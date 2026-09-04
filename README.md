# DSv4 native MXFP4 rollout — review snapshot

This branch exists only so the change can be read as a diff. It is **not**
mergeable and carries no history from the fork it was taken from: the base
commit holds the untouched files, the second commit holds the change.

## What it does

Lets verl train DeepSeek-V4-Flash-0731 with SGLang serving the routed experts in
the checkpoint's own packed MXFP4 layout, with no offline weight conversion.
Training stays bf16 throughout; only the rollout side holds 4-bit experts.

## Why it was needed

The Megatron bridge already re-packs routed experts to MXFP4 on export, and
SGLang can already serve them (`Mxfp4FlashinferCutlassMoEMethod`, the SM90 W4A16
CUTLASS path). The two could not be connected because of what happens in
between: SGLang's `process_weights_after_loading` runs the CUTLASS interleave
once at model load and republishes the four expert parameters as brand-new
`Parameter` objects. Those carry none of the attributes `set_weight_attrs`
attached, so every weight sync died in `deepseek_v4.py` at
`weight_loader = param.weight_loader` with

    AttributeError: 'Parameter' object has no attribute 'weight_loader'

on all 24 workers.

## How

`mxfp4_loader.py` registers as an SGLang `--custom-weight-loader` and runs a
stage -> load -> replay -> fold-back cycle inside every TP worker. It is the
SGLang counterpart of `verl/utils/vllm/vllm_fp4_utils.py`, which solves the same
problem for vLLM; the file docstring covers where the two differ.

Because verl posts a full sync as a series of ~512 MB buckets and the loader is
invoked once per bucket, staging is lazy and the sender appends a zero-length
`__mxfp4_end__` tensor to the final bucket to trigger the replay — the same
shape as the `__delta_spec__` protocol already used by `delta_loader.py`.

No SGLang patch or fork is required.

## Out of tree

Two files this change depends on do not live in this repo and are quoted in the
pull request description instead:

- `megatron/bridge/models/deepseek/deepseek_v4_bridge.py` — a fallback export
  path for `SGLANG_DSV4_FP4_DEQUANT=1`, inert on the MXFP4 route.
- `script/lib/train_dsv4_general_rl.sh` — registers the loader at server launch.

## Verification

Measured on 4x8 H200, DeepSeek-V4-Flash-0731 read directly, TP8/EP8 rollout,
Megatron TP1/PP4/EP8:

| | rollout experts | probs_diff | k3_kl | update_weights |
|---|---|---|---|---|
| offline-converted FP8 checkpoint | 8-bit blockwise | 0.00849 | 0.00156 | — |
| 0731, `DEQUANT=1` | 8-bit blockwise | 0.01082 | 0.00277 | 66.1 s |
| 0731, native MXFP4 (this change) | 4-bit packed | 0.01070 | 0.00300 | 85.2 s |

Sane divergence is the load-bearing evidence: staging or replay getting the
layout wrong would corrupt the experts and put `probs_diff` in the 0.5 range
rather than at 0.0107. The sync is also ~29% slower, which is the cost of
replaying the interleave across every MoE layer on each sync.

A CPU-only test of the cycle against a stand-in quant method confirms the three
properties the design depends on: parameter addresses survive (so a captured
CUDA graph stays valid), the folded-back values equal the replayed interleave,
and no staging state is left behind.
