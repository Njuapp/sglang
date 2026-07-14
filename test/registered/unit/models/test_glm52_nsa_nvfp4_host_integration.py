import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from transformers import PretrainedConfig

from sglang.srt import server_args as server_args_module
from sglang.srt.layers.attention import nsa_backend as nsa_backend_module
from sglang.srt.layers.attention.nsa.nsa_indexer import _to_2d_context_lens
from sglang.srt.layers.attention.nsa_backend import (
    NativeSparseAttnBackend,
)
from sglang.srt.mem_cache import memory_pool as memory_pool_module
from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool
from sglang.srt.model_executor import (
    model_runner_kv_cache_mixin as kv_cache_mixin_module,
)
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)
from sglang.srt.models.deepseek_common.index_cache import (
    get_nsa_index_cache_policy,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


def _glm52_hf_config() -> PretrainedConfig:
    return PretrainedConfig(
        architectures=["GlmMoeDsaForCausalLM"],
        index_topk=2048,
        index_head_dim=128,
        index_n_heads=32,
        index_topk_freq=4,
        index_skip_topk_offset=3,
        num_hidden_layers=78,
    )


class _HostModelRunner(ModelRunnerKVCacheMixin):
    pass


class TestGLM52NSANVFP4HostIntegration(unittest.TestCase):
    def test_deep_gemm_context_lens_use_2d_abi(self):
        one_dimensional = torch.tensor([64, 128, 256], dtype=torch.int32)
        normalized = _to_2d_context_lens(one_dimensional)
        self.assertEqual(tuple(normalized.shape), (3, 1))
        self.assertEqual(normalized.data_ptr(), one_dimensional.data_ptr())

        already_normalized = normalized.clone()
        self.assertIs(_to_2d_context_lens(already_normalized), already_normalized)

        speculative = torch.tensor([[64, 65], [128, 129]], dtype=torch.int32)
        torch.testing.assert_close(
            _to_2d_context_lens(speculative),
            speculative.reshape(-1, 1),
            rtol=0,
            atol=0,
        )

    def test_server_pool_and_native_decode_dispatch(self):
        hf_config = _glm52_hf_config()
        model_config_stub = SimpleNamespace(hf_config=hf_config)
        # "dummy" short-circuits weight/config discovery in __post_init__; the
        # production GLM-specific adjustment is invoked explicitly below.
        args = ServerArgs(
            model_path="dummy",
            device="cpu",
            tp_size=8,
            kv_cache_dtype="fp4_e2m1",
            fp4_kv_cache_recipe="nvfp4",
            nsa_decode_backend="flashmla_kv",
        )
        dense_threshold = SimpleNamespace(
            is_set=lambda: False,
            set=MagicMock(),
            get=lambda: hf_config.index_topk,
        )

        with (
            patch.object(
                ServerArgs,
                "get_model_config",
                return_value=model_config_stub,
            ),
            patch.object(
                server_args_module,
                "is_blackwell_supported",
                return_value=False,
            ),
            patch.object(server_args_module, "is_npu", return_value=False),
            patch.object(server_args_module, "is_hip", return_value=False),
            patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
            patch.object(
                server_args_module.envs,
                "SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD",
                dense_threshold,
            ),
        ):
            args._handle_model_specific_adjustments()

        self.assertEqual(args.attention_backend, "nsa")
        self.assertEqual(args.page_size, 64)
        self.assertEqual(args.kv_cache_dtype, "fp4_e2m1")
        self.assertEqual(args.nsa_prefill_backend, "flashmla_sparse")
        self.assertEqual(args.nsa_decode_backend, "flashmla_kv")
        dense_threshold.set.assert_called_once_with(hf_config.index_topk)

        runner = _HostModelRunner()
        runner.server_args = args
        runner.model_config = SimpleNamespace(
            hf_config=hf_config,
            kv_lora_rank=512,
            qk_nope_head_dim=192,
            qk_rope_head_dim=64,
            num_attention_heads=64,
            context_len=8192,
        )
        runner.req_to_token_pool = None
        runner.token_to_kv_pool = None
        runner.token_to_kv_pool_allocator = None
        runner.max_running_requests = 1
        runner.max_total_num_tokens = 64
        runner.num_effective_layers = 3
        runner.start_layer = 2
        runner.end_layer = 5
        runner.page_size = args.page_size
        runner.kv_cache_dtype = torch.uint8
        runner.device = "cpu"
        runner.use_mla_backend = True
        runner.enable_hisparse = False
        runner.is_hybrid_swa = False
        runner.is_draft_worker = False
        runner.hybrid_gdn_config = None
        runner.mambaish_config = None

        def model_layer(layer_id, **scale_fields):
            return SimpleNamespace(
                self_attn=SimpleNamespace(
                    attn_mqa=SimpleNamespace(layer_id=layer_id, **scale_fields)
                )
            )

        language_model = SimpleNamespace(
            layers=[
                model_layer(1, k_scale_float=0.125),
                model_layer(2, k_scale_float=0.25, k_scale=torch.tensor([9.0])),
                model_layer(3, k_scale=torch.tensor([0.5])),
                model_layer(4),
                model_layer(5, k_scale_float=0.75),
            ]
        )
        # Exercise the production wrapper.model unwrapping path rather than
        # replacing resolve_language_model with a test-only identity function.
        runner.model = SimpleNamespace(model=language_model)

        scale_set_calls = []
        original_scale_setter = NSATokenToKVPool.set_mla_kv_global_scale

        def scale_setter_spy(pool, layer_id, scale):
            scale_set_calls.append((layer_id, scale))
            return original_scale_setter(pool, layer_id, scale)

        saver_scope = SimpleNamespace(depth=0, ones_depths=[])

        @contextmanager
        def kv_region(_memory_type):
            saver_scope.depth += 1
            try:
                yield
            finally:
                saver_scope.depth -= 1

        fake_memory_saver = SimpleNamespace(region=kv_region)
        original_ones = torch.ones

        def record_ones(*args, **kwargs):
            saver_scope.ones_depths.append(saver_scope.depth)
            return original_ones(*args, **kwargs)

        with (
            patch.object(
                kv_cache_mixin_module,
                "is_float4_e2m1fn_x2",
                return_value=True,
            ),
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
            patch.object(
                memory_pool_module.TorchMemorySaverAdapter,
                "create",
                return_value=fake_memory_saver,
            ),
            patch.object(torch, "ones", side_effect=record_ones),
            patch.object(
                NSATokenToKVPool,
                "set_mla_kv_global_scale",
                new=scale_setter_spy,
            ),
        ):
            runner._init_pools()

        self.assertIsInstance(runner.token_to_kv_pool, NSATokenToKVPool)
        self.assertTrue(runner.token_to_kv_pool.nsa_kv_cache_store_nvfp4)
        self.assertEqual(runner.token_to_kv_pool.kv_cache_dim, 416)
        self.assertEqual(
            tuple(runner.token_to_kv_pool.get_key_buffer(2).shape),
            (128, 1, 416),
        )
        self.assertEqual(
            runner.token_to_kv_pool.get_mla_kv_global_scale(2).item(), 0.25
        )
        self.assertEqual(runner.token_to_kv_pool.get_mla_kv_global_scale(3).item(), 0.5)
        self.assertEqual(runner.token_to_kv_pool.get_mla_kv_global_scale(4).item(), 1.0)
        self.assertEqual([layer_id for layer_id, _ in scale_set_calls], [2, 3, 4])
        self.assertEqual(scale_set_calls[-1][1], 1.0)
        self.assertEqual(saver_scope.ones_depths, [0])

        scale_storage_ptr = runner.token_to_kv_pool.mla_kv_global_scale.data_ptr()
        runner._load_nsa_nvfp4_global_scales()
        self.assertEqual(
            runner.token_to_kv_pool.mla_kv_global_scale.data_ptr(),
            scale_storage_ptr,
        )

        mtp_runner = _HostModelRunner()
        mtp_runner.server_args = args
        mtp_runner.model_config = runner.model_config
        mtp_runner.req_to_token_pool = runner.req_to_token_pool
        # NextN/draft runners share the target runner's request pool.
        mtp_runner.is_draft_worker = True
        mtp_runner.token_to_kv_pool = None
        mtp_runner.token_to_kv_pool_allocator = runner.token_to_kv_pool_allocator
        mtp_runner.max_running_requests = 1
        mtp_runner.max_total_num_tokens = 64
        mtp_runner.num_effective_layers = 1
        mtp_runner.start_layer = 0
        mtp_runner.end_layer = 1
        mtp_runner.page_size = args.page_size
        mtp_runner.kv_cache_dtype = torch.uint8
        mtp_runner.device = "cpu"
        mtp_runner.use_mla_backend = True
        mtp_runner.enable_hisparse = False
        mtp_runner.is_hybrid_swa = False
        mtp_runner.hybrid_gdn_config = None
        mtp_runner.mambaish_config = None
        mtp_runner.model = SimpleNamespace(
            model=SimpleNamespace(
                decoder=model_layer(0, k_scale_float=torch.tensor([0.75]))
            )
        )
        with (
            patch.object(
                kv_cache_mixin_module,
                "is_float4_e2m1fn_x2",
                return_value=True,
            ),
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
            mtp_runner._init_pools()

        self.assertEqual(
            mtp_runner.token_to_kv_pool.get_mla_kv_global_scale(0).item(), 0.75
        )
        mtp_scale_ptr = mtp_runner.token_to_kv_pool.mla_kv_global_scale.data_ptr()
        for invalid_scale in (
            0.0,
            -1.0,
            float("nan"),
            float("inf"),
            torch.tensor([1.0, 2.0]),
        ):
            with self.subTest(invalid_scale=invalid_scale):
                mtp_runner.model.model.decoder = model_layer(
                    0, k_scale_float=invalid_scale
                )
                with self.assertRaisesRegex(ValueError, "global scale"):
                    mtp_runner._load_nsa_nvfp4_global_scales()
                self.assertEqual(
                    mtp_runner.token_to_kv_pool.mla_kv_global_scale.data_ptr(),
                    mtp_scale_ptr,
                )
                self.assertEqual(
                    mtp_runner.token_to_kv_pool.get_mla_kv_global_scale(0).item(),
                    0.75,
                )

        with (
            patch.object(
                nsa_backend_module,
                "get_attention_tp_size",
                return_value=8,
            ),
            patch.object(
                nsa_backend_module,
                "_require_native_nvfp4_flashmla",
            ),
            patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
        ):
            backend = NativeSparseAttnBackend(runner)

        batch_size = 2
        layer = SimpleNamespace(
            is_cross_attention=False,
            layer_id=2,
            tp_q_head_num=8,
            v_head_dim=512,
            head_dim=576,
            scaling=0.125,
        )
        backend.forward_metadata = SimpleNamespace(
            nsa_cache_seqlens_int32=torch.full((batch_size,), 64, dtype=torch.int32),
            flashmla_metadata=SimpleNamespace(
                flashmla_metadata=torch.zeros((1, 8), dtype=torch.int32),
                num_splits=torch.zeros(batch_size + 1, dtype=torch.int32),
            ),
        )
        forward_batch = SimpleNamespace(
            token_to_kv_pool=runner.token_to_kv_pool,
            hisparse_coordinator=None,
        )
        q = torch.zeros((batch_size, 8 * 512), dtype=torch.bfloat16)
        q_rope = torch.zeros((batch_size, 8 * 64), dtype=torch.bfloat16)
        topk_indices = (
            torch.arange(hf_config.index_topk, dtype=torch.int32)
            .remainder(runner.max_total_num_tokens)
            .repeat(batch_size, 1)
        )
        expected = torch.zeros((batch_size, 1, 8, 512), dtype=torch.bfloat16)

        with (
            patch.object(
                nsa_backend_module.envs.SGLANG_NSA_FUSE_TOPK,
                "get",
                return_value=True,
            ),
            patch.object(
                nsa_backend_module,
                "concat_mla_absorb_q_general",
                side_effect=lambda q_nope, q_rope: torch.cat((q_nope, q_rope), dim=-1),
            ),
            patch(
                "sgl_kernel.flash_mla.flash_mla_with_kvcache_nvfp4",
                return_value=(expected, torch.empty(0)),
                create=True,
            ) as native_fwd,
        ):
            actual = backend.forward_decode(
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
        native_fwd.assert_called_once()
        kwargs = native_fwd.call_args.kwargs
        self.assertEqual(tuple(kwargs["k_cache"].shape), (2, 64, 1, 416))
        self.assertEqual(tuple(kwargs["indices"].shape), (2, 1, 2048))
        torch.testing.assert_close(
            kwargs["indices"][:, 0], topk_indices, rtol=0, atol=0
        )
        self.assertEqual(
            kwargs["kv_global_scale"].data_ptr(),
            runner.token_to_kv_pool.get_mla_kv_global_scale(2).data_ptr(),
        )

    def test_layer_topk_dataflow(self):
        from sglang.srt.models import deepseek_v2 as deepseek_v2_module
        from sglang.srt.models.deepseek_v2 import DeepseekV2Model

        hf_config = _glm52_hf_config()
        shared_topk = torch.tensor([[7, 11]], dtype=torch.int32)
        observed_layer_inputs = []

        class LayerStub:
            def __init__(self, layer_id):
                self.skip_topk, self.next_skip_topk = get_nsa_index_cache_policy(
                    hf_config, layer_id, is_nextn=False
                )

            def __call__(
                self,
                positions,
                hidden_states,
                forward_batch,
                residual,
                *args,
                prev_topk_indices=None,
                **kwargs,
            ):
                observed_layer_inputs.append(prev_topk_indices)
                if self.skip_topk:
                    self_test.assertIs(prev_topk_indices, shared_topk)
                output_topk = prev_topk_indices if self.skip_topk else shared_topk
                if not self.next_skip_topk:
                    return hidden_states, residual
                return hidden_states, residual, output_topk

        self_test = self
        model = SimpleNamespace(
            pp_group=SimpleNamespace(is_first_rank=True, is_last_rank=True),
            start_layer=0,
            end_layer=6,
            first_k_dense_replace=100,
            layers=[LayerStub(i) for i in range(6)],
            layers_to_capture=[],
            enable_a2a_moe=False,
            llama_4_scaling_config=None,
        )
        forward_batch = SimpleNamespace(
            can_run_tbo=False,
            forward_mode=SimpleNamespace(is_idle=lambda: True),
        )
        hidden_states = torch.ones((2, 4), dtype=torch.bfloat16)

        with (
            patch.object(deepseek_v2_module, "nsa_use_prefill_cp", return_value=False),
            patch.object(
                deepseek_v2_module,
                "get_global_server_args",
                return_value=SimpleNamespace(disable_piecewise_cuda_graph=False),
            ),
        ):
            output = DeepseekV2Model.forward(
                model,
                input_ids=torch.zeros(2, dtype=torch.int64),
                positions=torch.arange(2),
                forward_batch=forward_batch,
                input_embeds=hidden_states,
            )

        self.assertIs(output, hidden_states)
        self.assertEqual(len(observed_layer_inputs), 6)
        self.assertIsNone(observed_layer_inputs[0])
        self.assertIsNone(observed_layer_inputs[1])
        self.assertIsNone(observed_layer_inputs[2])
        self.assertIs(observed_layer_inputs[3], shared_topk)
        self.assertIs(observed_layer_inputs[4], shared_topk)
        self.assertIs(observed_layer_inputs[5], shared_topk)


if __name__ == "__main__":
    unittest.main()
