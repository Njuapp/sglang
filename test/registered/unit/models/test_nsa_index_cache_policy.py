import unittest
from types import SimpleNamespace

from sglang.srt.models.deepseek_common.index_cache import (
    get_nsa_index_cache_policy,
    should_compute_nsa_indexer,
    validate_nsa_index_cache_config,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


class TestNSAIndexCachePolicy(unittest.TestCase):
    def test_glm52_offset_schedule(self):
        config = SimpleNamespace(index_topk_freq=4, index_skip_topk_offset=3)

        policies = [
            get_nsa_index_cache_policy(config, layer_id, is_nextn=False)
            for layer_id in range(78)
        ]

        self.assertEqual(
            policies[:10],
            [
                (False, False),
                (False, False),
                (False, True),
                (True, True),
                (True, True),
                (True, False),
                (False, True),
                (True, True),
                (True, True),
                (True, False),
            ],
        )
        self.assertEqual(
            [layer_id for layer_id, (skip, _) in enumerate(policies) if not skip],
            [0, 1, 2, *range(6, 78, 4)],
        )
        self.assertEqual(policies[-1], (True, False))

    def test_legacy_config_computes_every_layer(self):
        for config in (
            SimpleNamespace(),
            SimpleNamespace(index_topk_freq=None),
            SimpleNamespace(index_topk_freq=0),
        ):
            with self.subTest(index_topk_freq=getattr(config, "index_topk_freq", 1)):
                self.assertEqual(
                    [
                        get_nsa_index_cache_policy(config, layer_id, is_nextn=False)
                        for layer_id in range(4)
                    ],
                    [(False, False)] * 4,
                )

    def test_explicit_pattern_and_nextn(self):
        config = SimpleNamespace(index_topk_pattern="FFSF")
        self.assertEqual(
            [
                get_nsa_index_cache_policy(config, layer_id, is_nextn=False)
                for layer_id in range(4)
            ],
            [(False, False), (False, True), (True, False), (False, False)],
        )
        self.assertEqual(
            get_nsa_index_cache_policy(config, layer_id=78, is_nextn=True),
            (False, False),
        )

    def test_nonpositive_offset_is_rejected(self):
        config = SimpleNamespace(index_topk_freq=4, index_skip_topk_offset=0)
        with self.assertRaisesRegex(ValueError, "must be a positive integer"):
            get_nsa_index_cache_policy(config, layer_id=0, is_nextn=False)

    def test_invalid_explicit_patterns_are_rejected(self):
        invalid_configs = (
            SimpleNamespace(index_topk_pattern="SFFF"),
            SimpleNamespace(index_topk_pattern="FFXF"),
            SimpleNamespace(index_topk_pattern="FF", num_hidden_layers=4),
        )
        for config in invalid_configs:
            with self.subTest(pattern=config.index_topk_pattern):
                with self.assertRaises(ValueError):
                    validate_nsa_index_cache_config(config)

    def test_shared_layer_never_falls_back_to_unloaded_indexer(self):
        with self.assertRaisesRegex(RuntimeError, "no result was carried"):
            should_compute_nsa_indexer(
                skip_topk=True, is_nextn=False, prev_topk_indices=None
            )
        self.assertTrue(
            should_compute_nsa_indexer(
                skip_topk=True, is_nextn=True, prev_topk_indices=None
            )
        )
        self.assertFalse(
            should_compute_nsa_indexer(
                skip_topk=True, is_nextn=True, prev_topk_indices=object()
            )
        )


if __name__ == "__main__":
    unittest.main()
