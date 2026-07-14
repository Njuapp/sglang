import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt import server_args as server_args_module
from sglang.srt.layers.attention import nsa_backend as nsa_backend_module
from sglang.srt.layers.attention.nsa_backend import NativeSparseAttnBackend
from sglang.srt.mem_cache import memory_pool as memory_pool_module
from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool
from sglang.srt.model_executor import (
    model_runner_kv_cache_mixin as kv_cache_mixin_module,
)
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


class TestNSANVFP4MemoryAccounting(unittest.TestCase):
    def test_flashmla_kv_backend_dispatches_raw_cache_and_global_scale(self):
        backend = object.__new__(NativeSparseAttnBackend)
        backend.real_page_size = 64
        backend.kv_cache_dim = 416
        backend.nsa_kv_cache_store_nvfp4 = True
        backend.nsa_kv_cache_store_fp8 = False
        backend.nsa_index_topk = 64

        q = torch.zeros((2, 8, 576), dtype=torch.bfloat16)
        cache = torch.zeros((128, 1, 416), dtype=torch.uint8)
        global_scale = torch.tensor([1.25], dtype=torch.float32)
        page_table = torch.zeros((2, 64), dtype=torch.int32)
        layer = SimpleNamespace(tp_q_head_num=8, head_dim=576)
        flashmla_metadata = torch.zeros((1, 8), dtype=torch.int32)
        num_splits = torch.zeros(3, dtype=torch.int32)
        metadata = SimpleNamespace(
            nsa_cache_seqlens_int32=torch.full((2,), 64, dtype=torch.int32),
            flashmla_metadata=SimpleNamespace(
                flashmla_metadata=flashmla_metadata,
                num_splits=num_splits,
            ),
        )
        expected = torch.zeros((2, 1, 8, 512), dtype=torch.bfloat16)

        with patch(
            "sgl_kernel.flash_mla.flash_mla_with_kvcache_nvfp4",
            return_value=(expected, torch.empty(0)),
            create=True,
        ) as native_fwd:
            actual = backend._forward_flashmla_kv(
                q_all=q,
                kv_cache=cache,
                kv_global_scale=global_scale,
                v_head_dim=512,
                sm_scale=0.125,
                layer=layer,
                metadata=metadata,
                page_table_1=page_table,
            )

        self.assertIs(actual, expected)
        kwargs = native_fwd.call_args.kwargs
        self.assertIs(kwargs["kv_global_scale"], global_scale)
        self.assertEqual(tuple(kwargs["k_cache"].shape), (2, 64, 1, 416))
        self.assertEqual(tuple(kwargs["indices"].shape), (2, 1, 64))
        self.assertIs(kwargs["tile_scheduler_metadata"], flashmla_metadata)
        self.assertIs(kwargs["num_splits"], num_splits)

    def test_fa3_fallback_keeps_negative_and_positive_oob_indices_invalid(self):
        capacity = 5
        cache = torch.zeros((capacity, 1, 416), dtype=torch.uint8)
        page_table = torch.tensor([[0, -1, 4, capacity]], dtype=torch.int32)
        pool = SimpleNamespace(
            get_mla_kv_global_scale=lambda _layer_id: torch.tensor([1.0])
        )

        dequantized, remapped = NativeSparseAttnBackend._prepare_nvfp4_paged_fallback(
            object(),
            kv_cache=cache,
            page_table=page_table,
            layer_id=0,
            token_to_kv_pool=pool,
        )

        self.assertEqual(tuple(dequantized.shape), (4, 1, 576))
        torch.testing.assert_close(
            remapped,
            torch.tensor([[0, -1, 2, -1]], dtype=torch.int32),
            rtol=0,
            atol=0,
        )

    def test_paged_flashmla_sparse_never_receives_raw_nvfp4_rows(self):
        backend = object.__new__(NativeSparseAttnBackend)
        backend.forward_metadata = SimpleNamespace()
        backend.use_mha = False
        backend.nsa_prefill_impl = "flashmla_sparse"
        backend.nsa_decode_impl = "flashmla_sparse"
        backend.nsa_kv_cache_store_nvfp4 = True
        backend.nsa_kv_cache_store_fp8 = False

        raw_cache = torch.zeros((128, 1, 416), dtype=torch.uint8)
        dequantized_cache = torch.zeros((128, 1, 576), dtype=torch.bfloat16)
        physical_indices = torch.arange(128, dtype=torch.int32).view(2, 64)
        local_indices = physical_indices.clone()
        expected = torch.zeros((2, 8, 512), dtype=torch.bfloat16)
        pool = SimpleNamespace(get_key_buffer=lambda _layer_id: raw_cache)
        # This shape of extend uses the paged transform (as verify/speculative
        # paths do) while still selecting the sparse BF16 implementation.
        forward_mode = SimpleNamespace(
            is_target_verify=lambda: False,
            is_draft_extend=lambda include_v2=False: False,
            is_extend_without_speculative=lambda: False,
        )
        forward_batch = SimpleNamespace(
            forward_mode=forward_mode,
            token_to_kv_pool=pool,
            hisparse_coordinator=None,
        )
        layer = SimpleNamespace(
            is_cross_attention=False,
            layer_id=0,
            tp_q_head_num=8,
            v_head_dim=512,
            head_dim=576,
            scaling=0.125,
        )
        q = torch.zeros((2, 8 * 512), dtype=torch.bfloat16)
        q_rope = torch.zeros((2, 8 * 64), dtype=torch.bfloat16)

        with (
            patch.object(
                nsa_backend_module.envs.SGLANG_NSA_FUSE_TOPK,
                "get",
                return_value=True,
            ),
            patch.object(
                backend,
                "_prepare_nvfp4_paged_fallback",
                return_value=(dequantized_cache, local_indices),
            ) as prepare,
            patch.object(
                backend,
                "_forward_flashmla_sparse",
                return_value=expected,
            ) as sparse_fwd,
            patch.object(
                nsa_backend_module,
                "concat_mla_absorb_q_general",
                side_effect=lambda q_nope, rope: torch.cat([q_nope, rope], dim=-1),
            ),
        ):
            actual = backend.forward_extend(
                q=q,
                k=None,
                v=None,
                layer=layer,
                forward_batch=forward_batch,
                save_kv_cache=False,
                q_rope=q_rope,
                topk_indices=physical_indices,
            )

        self.assertIs(actual, expected)
        prepare.assert_called_once_with(
            kv_cache=raw_cache,
            page_table=physical_indices,
            layer_id=0,
            token_to_kv_pool=pool,
        )
        self.assertIs(sparse_fwd.call_args.kwargs["kv_cache"], dequantized_cache)
        self.assertIs(sparse_fwd.call_args.kwargs["page_table_1"], local_indices)

    def test_ragged_prefill_dequantizes_the_flattened_cached_prefix(self):
        backend = object.__new__(NativeSparseAttnBackend)
        physical_prefix = torch.tensor([7, 3, 9], dtype=torch.int32)
        backend.forward_metadata = SimpleNamespace(
            page_table_1_flattened=physical_prefix,
        )
        backend.use_mha = False
        backend.nsa_prefill_impl = "flashmla_sparse"
        backend.nsa_decode_impl = "flashmla_kv"
        backend.nsa_kv_cache_store_nvfp4 = True
        backend.nsa_kv_cache_store_fp8 = False

        raw_cache = torch.zeros((128, 1, 416), dtype=torch.uint8)
        dequantized_prefix = torch.zeros((3, 1, 576), dtype=torch.bfloat16)
        global_scale = torch.tensor([0.5], dtype=torch.float32)
        topk_indices = torch.arange(128, dtype=torch.int32).view(2, 64)
        expected = torch.zeros((2, 8, 512), dtype=torch.bfloat16)
        pool = SimpleNamespace(
            get_key_buffer=lambda _layer_id: raw_cache,
            get_mla_kv_global_scale=lambda _layer_id: global_scale,
        )
        forward_mode = SimpleNamespace(
            is_target_verify=lambda: False,
            is_draft_extend=lambda include_v2=False: False,
            is_extend_without_speculative=lambda: True,
        )
        forward_batch = SimpleNamespace(
            forward_mode=forward_mode,
            token_to_kv_pool=pool,
            extend_prefix_lens_cpu=[3],
            hisparse_coordinator=None,
        )
        layer = SimpleNamespace(
            is_cross_attention=False,
            layer_id=4,
            tp_q_head_num=8,
            v_head_dim=512,
            head_dim=576,
            scaling=0.125,
        )
        q = torch.zeros((2, 8 * 512), dtype=torch.bfloat16)
        q_rope = torch.zeros((2, 8 * 64), dtype=torch.bfloat16)

        with (
            patch.object(
                nsa_backend_module.envs.SGLANG_NSA_FUSE_TOPK,
                "get",
                return_value=True,
            ),
            patch.object(
                nsa_backend_module,
                "dequantize_nvfp4_k_cache_paged",
                return_value=dequantized_prefix,
            ) as dequantize,
            patch.object(
                backend,
                "_forward_flashmla_sparse",
                return_value=expected,
            ) as sparse_fwd,
            patch.object(
                nsa_backend_module,
                "concat_mla_absorb_q_general",
                side_effect=lambda q_nope, rope: torch.cat([q_nope, rope], dim=-1),
            ),
        ):
            actual = backend.forward_extend(
                q=q,
                k=None,
                v=None,
                layer=layer,
                forward_batch=forward_batch,
                save_kv_cache=False,
                q_rope=q_rope,
                topk_indices=topk_indices,
            )

        self.assertIs(actual, expected)
        dequantize.assert_called_once_with(
            raw_cache,
            physical_prefix,
            global_scale,
        )
        self.assertIs(sparse_fwd.call_args.kwargs["kv_cache"], dequantized_prefix)
        self.assertIs(sparse_fwd.call_args.kwargs["page_table_1"], topk_indices)

    def test_cpu_pool_uses_416_byte_main_rows_and_132_byte_index_rows(self):
        # The production dtype is CUDA-only.  For this host allocation test,
        # force the dtype predicate while using uint8, which is also the actual
        # physical storage dtype of the mixed NVFP4 row.
        size = 64
        page_size = 64
        layer_num = 2
        with (
            patch.object(
                memory_pool_module,
                "is_float4_e2m1fn_x2",
                return_value=True,
            ),
            patch.object(
                memory_pool_module,
                "maybe_init_custom_mem_pool",
                return_value=(False, None, None),
            ),
            patch.object(memory_pool_module, "_is_hip", False),
        ):
            pool = NSATokenToKVPool(
                size=size,
                page_size=page_size,
                kv_lora_rank=512,
                dtype=torch.uint8,
                qk_rope_head_dim=64,
                layer_num=layer_num,
                device="cpu",
                index_head_dim=128,
                enable_memory_saver=False,
                kv_cache_dim=416,
            )

        main_bytes_per_token = 416
        index_bytes_per_token = 128 + 4
        padded_main_tokens = size + page_size
        index_pages = (size + page_size + 1) // page_size

        self.assertTrue(pool.nsa_kv_cache_store_nvfp4)
        self.assertEqual(pool.kv_cache_dim, main_bytes_per_token)
        self.assertEqual(
            tuple(pool.kv_buffer[0].shape),
            (padded_main_tokens, 1, main_bytes_per_token),
        )
        self.assertEqual(pool.kv_buffer[0][0].nbytes, main_bytes_per_token)
        self.assertEqual(
            tuple(pool.index_k_with_scale_buffer[0].shape),
            (index_pages, page_size * index_bytes_per_token),
        )

        _, _, index_page_bytes = pool.get_state_buf_infos()
        self.assertEqual(
            index_page_bytes,
            [page_size * index_bytes_per_token] * layer_num,
        )

        expected_total = layer_num * (
            padded_main_tokens * main_bytes_per_token
            + index_pages * page_size * index_bytes_per_token
        )
        # The global scale is one FP32 scalar per layer, not a per-token cost.
        expected_total += layer_num * torch.float32.itemsize
        self.assertEqual(pool.get_kv_size_bytes(), expected_total)

    def test_scheduler_accounts_for_548_bytes_per_token_per_layer(self):
        runner = SimpleNamespace(
            use_mla_backend=True,
            kv_cache_dtype=torch.uint8,
            model_config=SimpleNamespace(
                hf_config=object(),
                kv_lora_rank=512,
                qk_rope_head_dim=64,
            ),
        )
        num_layers = 61
        with (
            patch.object(
                kv_cache_mixin_module,
                "is_deepseek_nsa",
                return_value=True,
            ),
            patch.object(
                kv_cache_mixin_module,
                "is_float4_e2m1fn_x2",
                return_value=True,
            ),
            patch.object(
                kv_cache_mixin_module,
                "get_nsa_index_head_dim",
                return_value=128,
            ),
        ):
            cell_size = ModelRunnerKVCacheMixin.get_cell_size_per_token(
                runner, num_layers
            )
            physical_mla_dim = ModelRunnerKVCacheMixin.calculate_mla_kv_cache_dim(
                runner
            )

        main_bytes_per_token = 416
        index_bytes_per_token = 128 + 4
        self.assertEqual(physical_mla_dim, main_bytes_per_token)
        self.assertEqual(
            cell_size,
            (main_bytes_per_token + index_bytes_per_token) * num_layers,
        )


class TestNSANVFP4ServerArgsGate(unittest.TestCase):
    @staticmethod
    def _make_args(**overrides) -> ServerArgs:
        args = ServerArgs(model_path="dummy")
        for name, value in overrides.items():
            setattr(args, name, value)
        return args

    def test_nsa_fp4_dtype_requires_nvfp4_recipe(self):
        args = self._make_args(
            kv_cache_dtype="fp4_e2m1",
            fp4_kv_cache_recipe="mxfp4",
        )
        with self.assertRaisesRegex(ValueError, "fp4-kv-cache-recipe=nvfp4"):
            args._set_default_nsa_kv_cache_dtype(major=9, quantization="")

        args.fp4_kv_cache_recipe = "nvfp4"
        args._set_default_nsa_kv_cache_dtype(major=9, quantization="")
        self.assertEqual(args.kv_cache_dtype, "fp4_e2m1")

    def test_nsa_fp4_defaults_to_p0_backends(self):
        args = self._make_args(
            kv_cache_dtype="fp4_e2m1",
            nsa_prefill_backend=None,
            nsa_decode_backend=None,
        )
        with patch.object(server_args_module, "is_hip", return_value=False):
            args._set_default_nsa_backends(kv_cache_dtype="fp4_e2m1", major=9)

        self.assertEqual(args.nsa_prefill_backend, "flashmla_sparse")
        self.assertEqual(args.nsa_decode_backend, "fa3")

    def test_nsa_fp4_rejects_non_sm90(self):
        args = self._make_args(
            kv_cache_dtype="fp4_e2m1",
            nsa_prefill_backend=None,
            nsa_decode_backend=None,
        )
        with (
            patch.object(server_args_module, "is_hip", return_value=False),
            self.assertRaisesRegex(ValueError, "only SM90/Hopper"),
        ):
            args._set_default_nsa_backends(kv_cache_dtype="fp4_e2m1", major=10)

    def test_nsa_fp4_compatibility_accepts_fallback_and_native_decode(self):
        for decode_backend in ("fa3", "flashmla_kv"):
            with self.subTest(decode_backend=decode_backend):
                valid_args = self._make_args(
                    attention_backend="nsa",
                    kv_cache_dtype="fp4_e2m1",
                    fp4_kv_cache_recipe="nvfp4",
                    nsa_prefill_backend="flashmla_sparse",
                    nsa_decode_backend=decode_backend,
                )
                with (
                    patch.object(server_args_module, "is_cuda", return_value=True),
                    patch.object(ServerArgs, "use_mla_backend", return_value=True),
                ):
                    valid_args._handle_kv4_compatibility()

        invalid_backend_pairs = (
            ("flashmla_kv", "fa3"),
            ("flashmla_sparse", "tilelang"),
            (None, "fa3"),
        )
        for prefill_backend, decode_backend in invalid_backend_pairs:
            with self.subTest(
                prefill_backend=prefill_backend,
                decode_backend=decode_backend,
            ):
                args = self._make_args(
                    attention_backend="nsa",
                    kv_cache_dtype="fp4_e2m1",
                    fp4_kv_cache_recipe="nvfp4",
                    nsa_prefill_backend=prefill_backend,
                    nsa_decode_backend=decode_backend,
                )
                with (
                    patch.object(server_args_module, "is_cuda", return_value=True),
                    patch.object(ServerArgs, "use_mla_backend", return_value=True),
                    self.assertRaisesRegex(
                        ValueError,
                        "flashmla_sparse.*nsa-decode-backend in.*fa3,flashmla_kv",
                    ),
                ):
                    args._handle_kv4_compatibility()

    def test_nsa_fp4_compatibility_rejects_wrong_recipe_and_non_cuda(self):
        args = self._make_args(
            attention_backend="nsa",
            kv_cache_dtype="fp4_e2m1",
            fp4_kv_cache_recipe="mxfp4",
            nsa_prefill_backend="flashmla_sparse",
            nsa_decode_backend="fa3",
        )
        with (
            patch.object(server_args_module, "is_cuda", return_value=True),
            patch.object(ServerArgs, "use_mla_backend", return_value=True),
            self.assertRaisesRegex(ValueError, "fp4-kv-cache-recipe=nvfp4"),
        ):
            args._handle_kv4_compatibility()

        args.fp4_kv_cache_recipe = "nvfp4"
        with (
            patch.object(server_args_module, "is_cuda", return_value=False),
            patch.object(ServerArgs, "use_mla_backend", return_value=True),
            self.assertRaisesRegex(RuntimeError, "only supported on CUDA"),
        ):
            args._handle_kv4_compatibility()

    def test_nsa_fp4_compatibility_rejects_hisparse(self):
        args = self._make_args(
            attention_backend="nsa",
            kv_cache_dtype="fp4_e2m1",
            fp4_kv_cache_recipe="nvfp4",
            nsa_prefill_backend="flashmla_sparse",
            nsa_decode_backend="fa3",
            enable_hisparse=True,
        )
        with (
            patch.object(server_args_module, "is_cuda", return_value=True),
            patch.object(ServerArgs, "use_mla_backend", return_value=True),
            self.assertRaisesRegex(ValueError, "does not yet support.*hisparse"),
        ):
            args._handle_kv4_compatibility()


if __name__ == "__main__":
    unittest.main()
