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
"""FSDP2 + TP (``_StridedShard``) in the sharded delta export.

Under FSDP2+TP a colwise-parallel weight (q/k/v, gate/up, lm_head) carries
``(_StridedShard(0, split_factor=tp), Shard(0))``: TP cuts dim 0 first, then FSDP
cuts each TP chunk again. Because the two cuts nest, each rank still owns one
contiguous run of dim 0, so the placement is an ordinary ``BlockPlacement`` -- the
tests below pin that against ``distribute_tensor``'s own sharding and check the
delta the sharded path yields is bit-identical to a full-gather diff.

A ``split_factor`` that does not match the inner cut would interleave instead of
nest, leaving a local tensor that is not one block; FSDP2 never builds that, and
it must still be rejected loudly rather than silently mistranslated.

Placement geometry is pure math, so these run on gloo/CPU via ``mp.spawn`` and
need no accelerator.
"""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor
from torch.distributed.tensor.placement_types import _StridedShard

from verl.checkpoint_engine.delta_sync.sparse_gather import gather_slot_entries_to_rank0, shard_delta_indices
from verl.workers.engine.spec import ShardSpec, derive_dtensor_placement, translate_flat_indices

WORLD_SIZE = 4
FULL_SHAPE = (8, 6)
# Positions perturbed between the two "training steps"; spread so that every rank
# of every geometry below owns at least one of them.
PERTURBED = [(0, 0), (1, 5), (2, 3), (3, 1), (4, 4), (5, 0), (6, 2), (7, 5), (7, 0)]


def _steps():
    """A parameter before and after a step, differing only at PERTURBED."""
    torch.manual_seed(0)
    old = torch.randn(*FULL_SHAPE, dtype=torch.bfloat16)
    new = old.clone()
    for r, c in PERTURBED:
        new[r, c] += 1.0
    return old, new


def _assert_block_matches_local(dt, full, placements):
    """The derived placement must describe exactly what this rank holds."""
    place, _contributes, _group = derive_dtensor_placement(ShardSpec.from_param(dt))
    local = dt.to_local()
    slices = tuple(slice(int(o), int(o) + int(s)) for o, s in zip(place.global_offset, place.local_shape, strict=False))
    assert tuple(full[slices].shape) == tuple(local.shape), (
        f"{placements}: block shape {tuple(full[slices].shape)} != local shape {tuple(local.shape)}"
    )
    assert torch.equal(full[slices], local), f"{placements}: block contents differ from the local shard"
    return place


def _assert_delta_matches_full_gather(mesh, placements, old, new):
    """Run the sharded delta path and compare it to a full-tensor byte diff."""
    dt_old = distribute_tensor(old, mesh, list(placements))
    dt_new = distribute_tensor(new, mesh, list(placements))

    place = _assert_block_matches_local(dt_new, new, placements)
    _, contributes, group = derive_dtensor_placement(ShardSpec.from_param(dt_new))

    local_new = dt_new.to_local().contiguous().view(-1)
    local_old = dt_old.to_local().contiguous().view(-1)
    if contributes:
        lidx, lval = shard_delta_indices(local_new, local_old, 0)
        gidx = translate_flat_indices(lidx, place)
    else:
        gidx = torch.empty(0, dtype=torch.int64)
        lval = torch.empty(0, dtype=local_new.dtype)

    counts = torch.tensor([int(gidx.numel())], dtype=torch.int64)
    gathered = gather_slot_entries_to_rank0(gidx.to(torch.int64), lval, counts, group=group)

    full_old = dt_old.full_tensor().reshape(-1)
    full_new = dt_new.full_tensor().reshape(-1)
    ref_idx = (full_new.view(torch.int16) != full_old.view(torch.int16)).nonzero(as_tuple=False).view(-1)
    ref_val = full_new[ref_idx]

    if dist.get_rank() != 0:
        return
    ((got_idx, got_val),) = gathered
    order = torch.argsort(got_idx)
    got_idx, got_val = got_idx[order], got_val[order]
    assert got_idx.numel() == ref_idx.numel(), (
        f"{placements}: gathered {got_idx.numel()} changed elements, full-gather diff found {ref_idx.numel()}"
    )
    assert got_idx.unique().numel() == got_idx.numel(), f"{placements}: ranks reported overlapping positions"
    assert torch.equal(got_idx, ref_idx.to(got_idx.dtype)), f"{placements}: positions diverge"
    assert torch.equal(got_val.view(torch.int16), ref_val.view(torch.int16)), f"{placements}: values diverge"


def _supported_geometries_worker(rank, world_size, rendezvous_file):
    dist.init_process_group(backend="gloo", init_method=f"file://{rendezvous_file}", rank=rank, world_size=world_size)
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
        old, new = _steps()

        # colwise TP: both mesh dims cut dim 0, so FSDP's placement is strided
        colwise = (_StridedShard(0, split_factor=2), Shard(0))
        _assert_delta_matches_full_gather(mesh, colwise, old, new)
        place, _, _ = derive_dtensor_placement(ShardSpec.from_param(distribute_tensor(new, mesh, list(colwise))))
        assert place.is_flat_contiguous, (
            "a nested dim-0 cut leaves whole trailing dims, so it must keep the single-add fast path"
        )

        # rowwise TP: TP cuts dim 1, so FSDP stays a plain Shard(0) -- the control
        _assert_delta_matches_full_gather(mesh, (Shard(0), Shard(1)), old, new)
    finally:
        dist.destroy_process_group()


def _rejects_interleaved_worker(rank, world_size, rendezvous_file):
    dist.init_process_group(backend="gloo", init_method=f"file://{rendezvous_file}", rank=rank, world_size=world_size)
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))

        # split_factor larger than the inner cut: the pieces interleave.
        spec = ShardSpec(full_shape=FULL_SHAPE, mesh=mesh, placements=(_StridedShard(0, split_factor=3), Shard(0)))
        with pytest.raises(NotImplementedError, match="not one block"):
            derive_dtensor_placement(spec)

        # inner dim cuts a *different* tensor dim, so nothing nests with the stride.
        spec = ShardSpec(full_shape=FULL_SHAPE, mesh=mesh, placements=(_StridedShard(0, split_factor=2), Shard(1)))
        with pytest.raises(NotImplementedError, match="not one block"):
            derive_dtensor_placement(spec)
    finally:
        dist.destroy_process_group()


def test_fsdp2_tp_strided_shard_delta_matches_full_gather(tmp_path):
    mp.spawn(
        _supported_geometries_worker,
        args=(WORLD_SIZE, str(tmp_path / "strided_rdzv")),
        nprocs=WORLD_SIZE,
        join=True,
    )


def test_interleaved_strided_shard_is_rejected(tmp_path):
    mp.spawn(
        _rejects_interleaved_worker,
        args=(WORLD_SIZE, str(tmp_path / "reject_rdzv")),
        nprocs=WORLD_SIZE,
        join=True,
    )
