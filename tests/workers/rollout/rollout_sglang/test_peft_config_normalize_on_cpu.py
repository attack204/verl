# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
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

"""Unit tests for normalizing an engine's adapter config for SGLang's adapter loader.

`BaseEngine.get_per_tensor_param` declares its second return value `Optional[dict]`. The
fixtures here are built by the real producers rather than hand-written, so the tests
break if either producer changes shape.

The two producers disagree on `target_modules`: peft renders a set of concrete module
names, megatron sends the `"all-linear"` shorthand. Both have to reach SGLang in a form
it recognizes, which is what most of these tests pin down.
"""

from __future__ import annotations

import json

import pytest
from peft import LoraConfig, TaskType

from verl.utils.megatron_peft_utils import build_peft_config_for_vllm
from verl.workers.rollout.sglang_rollout.utils import normalize_peft_config_for_sglang

SEVEN_PROJECTIONS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _fsdp_shape() -> dict:
    """What verl/workers/engine/fsdp/transformer_impl.py returns."""
    return LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=SEVEN_PROJECTIONS,
        task_type=TaskType.CAUSAL_LM,
    ).to_dict()


def _megatron_shape() -> dict:
    """What verl/workers/engine/megatron/transformer_impl.py returns."""
    return build_peft_config_for_vllm({"rank": 32, "alpha": 32})


class TestNormalizePeftConfigForSGLang:
    def test_result_is_json_serializable(self):
        # The config crosses an HTTP boundary, so every value must survive json.dumps.
        json.dumps(normalize_peft_config_for_sglang(_fsdp_shape()))

    def test_enums_are_unwrapped_to_strings(self):
        result = normalize_peft_config_for_sglang(_fsdp_shape())
        assert result["task_type"] == "CAUSAL_LM"
        assert result["peft_type"] == "LORA"

    def test_target_modules_is_materialized_as_list(self):
        result = normalize_peft_config_for_sglang({"target_modules": {"q_proj", "v_proj"}, "peft_type": "LORA"})
        assert isinstance(result["target_modules"], list)
        assert sorted(result["target_modules"]) == ["q_proj", "v_proj"]

    def test_input_is_not_mutated(self):
        original = _fsdp_shape()
        before = dict(original)
        normalize_peft_config_for_sglang(original)
        assert original == before

    def test_megatron_shape_is_accepted(self):
        result = normalize_peft_config_for_sglang(_megatron_shape())
        assert result["peft_type"] == "LORA"
        assert result["task_type"] == "CAUSAL_LM"
        json.dumps(result)

    def test_all_linear_shorthand_survives_as_a_string(self):
        # SGLang recognizes "all-linear" and expands it against the served model, but
        # only while it is still a string. Materializing it as a list yields ten
        # one-letter entries, which SGLang reads as module names and then rejects the
        # adapter as incompatible with its LoRA memory pool.
        assert _megatron_shape()["target_modules"] == "all-linear"
        assert normalize_peft_config_for_sglang(_megatron_shape())["target_modules"] == "all-linear"

    def test_missing_peft_type_is_rejected_not_guessed(self):
        # Neither producer omits peft_type today. Should a third shape appear, it must
        # fail with something a reader can act on rather than be silently filled in
        # with a plausible value.
        with pytest.raises(ValueError, match="peft_type"):
            normalize_peft_config_for_sglang({"target_modules": ["q_proj"], "r": 8})

    def test_rejection_lists_the_keys_present(self):
        with pytest.raises(ValueError) as excinfo:
            normalize_peft_config_for_sglang({"target_modules": ["q_proj"], "r": 8})
        # The key listing is there to orient a reader who has to go find the producer.
        assert "Keys present: r, target_modules" in str(excinfo.value)
