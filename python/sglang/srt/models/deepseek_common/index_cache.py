# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


def validate_nsa_index_cache_config(config) -> bool:
    """Validate NSA cross-layer top-k sharing and return whether it is enabled."""
    raw_index_topk_freq = getattr(config, "index_topk_freq", 1)
    # Older checkpoints sometimes encode the legacy "every layer" policy as
    # null or zero. Preserve that behavior while rejecting other invalid data.
    index_topk_freq = 1 if raw_index_topk_freq in (None, 0) else raw_index_topk_freq
    if not isinstance(index_topk_freq, int) or index_topk_freq <= 0:
        raise ValueError("index_topk_freq must be a non-negative integer")

    index_topk_pattern = getattr(config, "index_topk_pattern", None)
    index_skip_topk_offset = getattr(config, "index_skip_topk_offset", None)

    if index_topk_pattern is not None:
        if not isinstance(index_topk_pattern, str) or not index_topk_pattern:
            raise ValueError("index_topk_pattern must be a non-empty string")
        invalid = set(index_topk_pattern) - {"F", "S"}
        if invalid:
            raise ValueError(
                "index_topk_pattern may contain only 'F' and 'S', got "
                f"{sorted(invalid)}"
            )
        num_hidden_layers = getattr(config, "num_hidden_layers", None)
        if num_hidden_layers is not None and len(index_topk_pattern) < int(
            num_hidden_layers
        ):
            raise ValueError(
                "index_topk_pattern is shorter than num_hidden_layers: "
                f"{len(index_topk_pattern)} < {num_hidden_layers}"
            )
        if index_topk_pattern[0] == "S":
            raise ValueError(
                "index_topk_pattern cannot skip layer 0 because no prior top-k exists"
            )

    if index_skip_topk_offset is not None and (
        not isinstance(index_skip_topk_offset, int) or index_skip_topk_offset <= 0
    ):
        raise ValueError("index_skip_topk_offset must be a positive integer")

    return index_topk_freq > 1 or (
        index_topk_pattern is not None and "S" in index_topk_pattern
    )


def get_nsa_index_cache_policy(config, layer_id: int, is_nextn: bool):
    """Return whether this and the next NSA layer reuse indexer top-k indices."""
    validate_nsa_index_cache_config(config)
    if is_nextn:
        # Cross-iteration MTP reuse is deliberately outside this policy. A
        # NextN layer computes its own indexer result when speculative decoding
        # is enabled.
        return False, False

    index_topk_freq = getattr(config, "index_topk_freq", 1) or 1
    index_topk_pattern = getattr(config, "index_topk_pattern", None)
    index_skip_topk_offset = getattr(config, "index_skip_topk_offset", None)

    if index_topk_pattern is not None:
        skip_topk = index_topk_pattern[layer_id] == "S"
        next_skip_topk = (
            index_topk_pattern[layer_id + 1] == "S"
            if layer_id < len(index_topk_pattern) - 1
            else False
        )
        return skip_topk, next_skip_topk

    if index_skip_topk_offset is not None:
        skip_topk = max(layer_id - index_skip_topk_offset + 1, 0) % index_topk_freq != 0
        next_skip_topk = (
            max(layer_id - index_skip_topk_offset + 2, 0) % index_topk_freq != 0
        )
        return skip_topk, next_skip_topk

    return (
        max(layer_id - 1, 0) % index_topk_freq != 0,
        layer_id % index_topk_freq != 0,
    )


def should_compute_nsa_indexer(
    skip_topk: bool, is_nextn: bool, prev_topk_indices
) -> bool:
    """Whether this layer must run its own indexer."""
    if skip_topk and prev_topk_indices is None and not is_nextn:
        raise RuntimeError(
            "NSA layer is configured to reuse top-k indices, but no result was "
            "carried from the preceding layer"
        )
    return not skip_topk or (is_nextn and prev_topk_indices is None)
