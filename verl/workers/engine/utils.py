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

import os
import random

import numpy as np
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.device import is_npu_available
from verl.utils.device import manual_seed as device_manual_seed
from verl.utils.device import manual_seed_all as device_manual_seed_all
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import rearrange_micro_batches, restore_dynamic_batch


def enable_full_determinism(seed: int):
    """
    Helper function for reproducibility in distributed training.
    See https://pytorch.org/docs/stable/notes/randomness.html for details.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
    os.environ["NCCL_DETERMINISTIC"] = "1"
    os.environ["NCCL_ALGO"] = "Ring"
    os.environ["NCCL_PROTO"] = "Simple"
    # flash-attn's Triton cross-entropy kernel (used by logprobs_from_logits to
    # compute log_probs) has a non-deterministic reduction that is NOT covered by
    # FLASH_ATTENTION_DETERMINISTIC (only governs attention kernels' backward) nor
    # by torch.use_deterministic_algorithms (Triton custom ops don't trigger
    # warn_only). Force the pure-PyTorch log_softmax+gather path instead.
    os.environ.setdefault("VERL_DISABLE_FLASH_ATTN_CE", "1")
    if is_npu_available:
        # The environment variable required to enable deterministic mode on Ascend NPUs.
        os.environ["HCCL_DETERMINISTIC"] = "true"
        os.environ["CLOSE_MATMUL_K_SHIFT"] = "1"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device_manual_seed(seed)
    device_manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    # Enable CUDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def pad_packed_inputs(
    input_ids_rmpad: torch.Tensor,
    position_ids_rmpad: torch.Tensor | None,
    pad_size: int,
    pad_value: float = 0,
):
    """Right-pad a packed ``(1, total_nnz)`` batch by ``pad_size`` tokens.

    Mirrors the padding :func:`verl.utils.ulysses.ulysses_pad` applies to reach a multiple of the
    sequence-parallel size: the appended ``position_ids`` restart from 0, so the pad tokens form
    one trailing varlen segment instead of extending the last real sequence.
    """
    if pad_size <= 0:
        return input_ids_rmpad, position_ids_rmpad

    input_ids_rmpad = torch.nn.functional.pad(input_ids_rmpad, (0, pad_size), value=pad_value)
    if position_ids_rmpad is not None:
        pad_position_ids = torch.arange(
            pad_size, dtype=position_ids_rmpad.dtype, device=position_ids_rmpad.device
        ).unsqueeze(0)
        if position_ids_rmpad.dim() == 3:  # (rope_dim, 1, total_nnz) mRoPE layout
            pad_position_ids = pad_position_ids.unsqueeze(0).repeat(position_ids_rmpad.size(0), 1, 1)
        position_ids_rmpad = torch.cat((position_ids_rmpad, pad_position_ids), dim=-1)
    return input_ids_rmpad, position_ids_rmpad


def prepare_micro_batches(
    data: TensorDict,
    dp_group=None,
    num_batches_divided_by=None,
    same_micro_num_in_dp=True,
    min_num_micro_batch=None,
    use_dynamic_bsz_balance=True,
):
    """
    Prepare micro batches from data.
    """
    use_dynamic_bsz = tu.get_non_tensor_data(data=data, key="use_dynamic_bsz", default=True)
    sp_size = tu.get_non_tensor_data(data=data, key="sp_size", default=1)

    force_group_size = tu.get_non_tensor_data(data=data, key="force_group_size", default=1)

    if use_dynamic_bsz:
        assert "max_token_len_per_gpu" in data.keys(), "max_token_len_per_gpu must be set when use_dynamic_bsz is True"
        max_token_len_per_gpu = data["max_token_len_per_gpu"]
        max_token_len = max_token_len_per_gpu * sp_size
        micro_batches, batch_idx_list = rearrange_micro_batches(
            data,
            max_token_len=max_token_len,
            dp_group=dp_group,
            num_batches_divided_by=num_batches_divided_by,
            same_micro_num_in_dp=same_micro_num_in_dp,
            min_num_micro_batch=min_num_micro_batch,
            use_dynamic_bsz_balance=use_dynamic_bsz_balance,
            force_group_size=force_group_size,
        )
    else:
        total_data_size = len(data)
        micro_batch_size_per_gpu = data["micro_batch_size_per_gpu"]
        assert total_data_size % (force_group_size * micro_batch_size_per_gpu) == 0, (
            "data size must be divisible by force_group_size * micro_batch_size_per_gpu"
        )
        micro_batches = tu.chunk_tensordict(data, total_data_size // (micro_batch_size_per_gpu * force_group_size))
        batch_idx_list = None
    return micro_batches, batch_idx_list


def postprocess_batch_func(output_lst, indices, data: TensorDict):
    """postprocess the output of a forward_backward_batch.
    output_lst is a list of dict containing outputs for each micro-batch
    reorder entropy and outputs. Return None for other pp ranks
    only on last rank. It should be on every tp rank

    each losses_reduced contains 1. model_output, 2. loss, 3. metrics.
    """

    use_dynamic_bsz = tu.get_non_tensor_data(data=data, key="use_dynamic_bsz", default=True)
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    assert pad_mode == DatasetPadMode.NO_PADDING, "postprocess_batch_func only support NO_PADDING pad_mode"

    # losses_reduced is a list of dict containing outputs for each micro-batch
    # reorder entropy and outputs. Return None for other pp ranks
    # only on last rank. It should be on every tp rank

    # losses_reduced contains 1. model_output, 2. loss, 3. metrics.
    # We perform reverse

    model_output = {}
    losses = []
    aggregated_metrics = {}

    # model output
    for o in output_lst:
        if "model_output" in o:
            for key, val in o["model_output"].items():
                if key not in model_output:
                    model_output[key] = []
                model_output[key].append(val)

    # concat results from micro batches
    for key, val in model_output.items():
        if pad_mode == DatasetPadMode.NO_PADDING:
            tensors = [tensor for nt in model_output[key] for tensor in nt.unbind()]
            model_output[key] = torch.nested.as_nested_tensor(tensors, layout=torch.jagged)
        else:
            raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        # reverse with dynamic bsz
        if use_dynamic_bsz:
            model_output[key] = restore_dynamic_batch(model_output[key], indices)

    # loss
    for o in output_lst:
        if "loss" in o:
            losses.append(o["loss"])

    # metrics
    for o in output_lst:
        if "metrics" in o:
            metrics = o["metrics"]
            append_to_dict(aggregated_metrics, metrics)

    output = {
        "model_output": model_output,
        "loss": losses,
        "metrics": aggregated_metrics,
    }

    return output


# ---- sharded-delta HF export (backend side) --------------------------------
# The delta checkpoint engine consumes FINAL HF-coordinate deltas; everything
# backend-specific -- the weight->HF naming, the to-HF conversion, the diff and
# its snapshot -- happens here, on the backend side of the contract. The engine
# keeps only collectives, bucketing and the wire. This module holds the
# DTensor-generic pieces the backends share; the probe-driven machinery for
# opaque to_hf callables lives in the veomni backend's own utils.


def _prodshape(shape) -> int:
    n = 1
    for x in shape:
        n *= int(x)
    return n


def _hf_entry_identity(name, spec, place, lidx, lval):
    """Identity profile: the param IS its own single slot (weight name == HF name)
    -- translate the shard-local delta to within-param coordinates. int32
    positions: the wire is int32 anyway and the engine asserts the range."""
    from .spec import translate_flat_indices

    gidx = (translate_flat_indices(lidx, place) if lidx.numel() else lidx).to(torch.int32)
    counts = torch.zeros(1, dtype=torch.int64)
    counts[0] = int(gidx.numel())
    return [(name, tuple(spec.full_shape))], str(lval.dtype).replace("torch.", ""), counts, gidx, lval


def _hf_entry_row_slots(name, spec, place, lidx, lval):
    """Dim-0 identity slot profile: the logical tensor's dim 0 enumerates HF
    tensors and the split copies values verbatim, so slot ``e`` IS ``full[e]``
    and a full-tensor position's slot is one divmod away. A fused expert stack
    ``(num_experts, *, *)`` is the case this exists for: torchtitan's adapter
    splits it into one HF weight per expert with no reshape and no transpose.

    Why not veomni's ``to_hf_chunk`` probe path: that one re-runs the backend's
    converter on every touched dim-0 row because the conversion is an opaque
    callable. Here it is known to be a slice, so the whole entry is a handful of
    vectorized ops -- which matters, since a 128-expert model has one such row
    per (layer, w1/w2/w3) and the probe path would be ~128 conversions each.

    Relies on ``translate_flat_indices`` being monotonic (it is: local flat order
    walks a block in row-major, which is increasing in full-tensor row-major
    order), so positions arrive already grouped by slot and one ``searchsorted``
    recovers the run lengths. Were that to stop holding, elements would land in a
    neighbouring expert, which is what the export test's byte-exact per-HF-tensor
    comparison fails on.
    """
    from .spec import translate_flat_indices

    slots = spec.hf_slots
    n_slots = len(slots)
    rows = int(spec.full_shape[0])
    assert n_slots == rows, (
        f"{name}: dim-0 identity slots need one slot per dim-0 row, got {n_slots} slots for {rows} rows; "
        "a converter that emits several HF tensors per row belongs on the to_hf_chunk path"
    )
    dtype_str = str(lval.dtype).replace("torch.", "")
    if lidx.numel() == 0:
        return slots, dtype_str, torch.zeros(n_slots, dtype=torch.int64), lidx.to(torch.int32), lval

    row_numel = max(_prodshape(spec.full_shape[1:]), 1)
    gidx = translate_flat_indices(lidx, place)
    edges = torch.arange(n_slots + 1, device=gidx.device, dtype=gidx.dtype) * row_numel
    # One D2H for the run lengths. The diff's nonzero already synced this stream,
    # and counts have to reach the host inside the gather anyway.
    counts = torch.searchsorted(gidx, edges).diff().cpu()
    return slots, dtype_str, counts, (gidx % row_numel).to(torch.int32), lval


def hf_delta_export(gen, snaps: dict, entry_fn):
    """STEADY export: wrap a raw ``(name, local_shard, spec)`` exporter into final
    HF-coordinate delta entries ``(slots, dtype_str, counts, hf_idx, hf_val,
    gather_group)`` -- diff against the pinned snapshot, refresh it, then hand the
    shard-local delta to ``entry_fn(name, spec, place, lidx, lval)``, the engine's
    per-param entry builder (FSDP: identity only; veomni adds the EP converter).
    The delta engine consumes these entries verbatim (batch -> gather -> wire); no
    spec, no placement and no conversion cross the boundary. Requires a prior seed
    pass."""
    from verl.checkpoint_engine.delta_sync.sparse_gather import shard_delta_indices

    from .spec import derive_dtensor_placement

    for name, local, spec in gen:
        local = local.detach().contiguous().view(-1)
        snap = snaps.get(name)
        assert snap is not None and snap.numel() == local.numel(), (
            f"{name}: no seed snapshot for this shard; run the seed export first"
        )
        if spec.place is not None:
            # explicit exporter override: the backend declared the whole triple
            # (hybrid geometries are not derivable from DTensor facts alone).
            place, contributes, pg = spec.place, spec.contributes, spec.gather_group
        else:
            place, contributes, pg = derive_dtensor_placement(spec)
        if contributes:
            base = snap.to(local.device, non_blocking=True)
            lidx, lval = shard_delta_indices(local, base, 0)
        else:
            # replicated param owned by another rank; empty delta keeps lockstep.
            lidx = torch.empty(0, dtype=torch.int64, device=local.device)
            lval = torch.empty(0, dtype=local.dtype, device=local.device)
        snap.copy_(local, non_blocking=True)
        yield (*entry_fn(name, spec, place, lidx, lval), pg)


def prime_delta_snapshots(gen, snaps: dict, pin: bool) -> None:
    """Snapshot each rank's current shards to CPU as the steady diff base. Run
    right after the seed's full-weight sync: weights do not move during the
    sync, so the snapshots equal exactly what the rollout side received.

    ``pin`` selects pinned vs pageable host memory and is the ENGINE's call
    (``BaseEngine.delta_pin_snapshots``): pinning a whole shard set
    (cudaHostAlloc) competes with everything else that pins on the node and its
    failure surfaces as a CUDA out-of-memory; pageable costs a slower H2D on
    the diff read-back but cannot OOM the device."""
    for name, local, _spec in gen:
        local = local.detach().contiguous().view(-1)
        snap = snaps.get(name)
        if snap is None or snap.numel() != local.numel():
            snap = torch.empty_like(local, device="cpu", pin_memory=pin)
            snaps[name] = snap
        snap.copy_(local, non_blocking=True)
