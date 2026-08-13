# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Both engines must answer ``get_per_tensor_param``'s peft_config the same shape.

The two producers drifted apart once already: the FSDP engine returns
``LoraConfig.to_dict()`` while megatron's ``build_peft_config_for_vllm()`` had no
``peft_type`` at all. Nothing caught it because the only consumer at the time was
vLLM's rollout, which does not read that key -- SGLang's adapter loader does, and
rejects a config without it.

Neither producer needs a GPU or a distributed context to build its dict, so the
agreement is checkable here, which is the point: a contract that is only exercised
by a two-node training run is a contract that drifts.
"""

import pytest

# The keys BaseEngine.get_per_tensor_param documents as required.
REQUIRED = ("peft_type", "task_type", "r", "lora_alpha", "target_modules", "lora_dropout", "bias")


def _megatron_config() -> dict:
    from verl.utils.megatron_peft_utils import build_peft_config_for_vllm

    return build_peft_config_for_vllm({"rank": 32, "alpha": 32, "dropout": 0.0})


def _fsdp_config() -> dict:
    from peft import LoraConfig, TaskType

    return LoraConfig(r=32, lora_alpha=32, target_modules=["q_proj"], task_type=TaskType.CAUSAL_LM).to_dict()


@pytest.mark.parametrize("producer", [_megatron_config, _fsdp_config], ids=["megatron", "fsdp"])
def test_required_keys_present(producer):
    cfg = producer()
    missing = [k for k in REQUIRED if k not in cfg]
    assert not missing, f"{producer.__name__} is missing {missing}"


@pytest.mark.parametrize("producer", [_megatron_config, _fsdp_config], ids=["megatron", "fsdp"])
def test_enum_valued_keys_are_enums(producer):
    """Documented as enum members, and consumers unwrap them on that basis."""
    from peft import PeftType, TaskType

    cfg = producer()
    assert isinstance(cfg["task_type"], TaskType), type(cfg["task_type"])
    assert isinstance(cfg["peft_type"], PeftType), type(cfg["peft_type"])


def test_producers_agree_on_the_required_subset():
    """Not a demand that the dicts be equal -- only that neither omits what the
    other supplies, which is the failure that actually happened."""
    mg, fs = set(_megatron_config()), set(_fsdp_config())
    only_fsdp = sorted((fs - mg) & set(REQUIRED))
    only_megatron = sorted((mg - fs) & set(REQUIRED))
    assert not only_fsdp, f"required keys only FSDP supplies: {only_fsdp}"
    assert not only_megatron, f"required keys only megatron supplies: {only_megatron}"


def test_sglang_normalizer_accepts_both():
    """The consumer that motivated this: it rejects a config with no peft_type."""
    from verl.workers.rollout.sglang_rollout.utils import normalize_peft_config_for_sglang

    for name, cfg in (("megatron", _megatron_config()), ("fsdp", _fsdp_config())):
        out = normalize_peft_config_for_sglang(cfg)
        assert isinstance(out["peft_type"], str), f"{name}: peft_type not unwrapped to str"
        assert isinstance(out["task_type"], str), f"{name}: task_type not unwrapped to str"
