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
"""Backend-agnostic sparse delta apply.

A sparse delta cannot be handed to an inference engine's weight loader directly:
the loader owns the checkpoint-name mapping, the tensor-parallel slicing and the
fused-projection splitting, and reimplementing any of that per engine would
duplicate the part most likely to drift. Instead each parameter's delta is
densified into a full-shape tensor whose *unchanged* positions hold NaN, the
engine's own ``load_weights`` is called with it, and the ``Tensor.copy_`` those
loaders ultimately land on is made to skip NaN positions. The engine keeps
resolving names and shards; only the final write is narrowed.

Two properties make that safe, and both have to be enforced rather than assumed:

* The masked semantics may only apply to writes that land in the model's own
  parameters and buffers. A loader that stages through a scratch tensor would
  otherwise have the NaNs replaced by whatever that scratch tensor happened to
  hold, and the resulting garbage -- no longer NaN -- would be copied into the
  parameter by the next write with nothing left to mask it.
* Nothing may leave NaN behind in a live parameter. Masking only fires for
  same-shape floating-point writes, so a loader that reshapes, transposes, casts
  through an integer dtype or derives values arithmetically falls back to an
  ordinary copy, and a NaN-bearing source then lands verbatim in the weights.
  Checking each write would cost a device sync per parameter, which is why the
  check is a single sweep at the end of the flush instead.

Post-load transforms (fp8 scales, MoE bias, MLA ``w_kc``/``w_vc``) must run
outside :func:`masked_copy`: they read the weights back and would compute against
half-written state. Callers are responsible for that scoping -- keep the context
around ``load_weights`` alone.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)


def decode_param(encoding: str, positions: torch.Tensor, values: torch.Tensor, param: dict) -> torch.Tensor:
    """Densify one parameter's sparse delta into a full-shape NaN-masked tensor.

    ``indices`` positions are reinterpreted through an int32 view (8 B/element
    transient) rather than a per-byte int64 unpack (32 B/element), so even a
    full-seed flush of a large embedding stays within a few GiB.
    """
    numel = math.prod(param["shape"])
    dtype = getattr(torch, param["dtype"])
    flat = torch.full((numel,), float("nan"), dtype=dtype, device=values.device)
    vals = values[param["val_start"] : param["val_end"]]
    if vals.numel() == 0:
        return flat.view(param["shape"])

    pos_bytes = positions[param["pos_start"] : param["pos_end"]]
    if encoding != "indices":
        raise ValueError(f"unsupported delta encoding: {encoding!r}")
    # pos_start is always int32-aligned (each entry packs nnz * 4 bytes).
    idx = pos_bytes.view(torch.int32).to(torch.int64)

    flat.index_copy_(0, idx, vals.to(dtype))
    return flat.view(param["shape"])


def _storage_ids(models) -> set[int]:
    """Storage addresses backing the models' parameters and buffers.

    A narrowed or reshaped view of a parameter shares its parameter's storage, so
    comparing storage addresses recognises the slices engine loaders actually
    write to, which comparing tensor addresses would not. Nothing here can
    collide with a scratch tensor: these storages are alive for the duration of
    the load, so the allocator cannot hand their addresses out again.
    """
    ids = set()
    for model in models:
        for tensor in list(model.parameters()) + list(model.buffers()):
            try:
                ids.add(tensor.untyped_storage().data_ptr())
            except Exception:  # meta / fake tensors have no storage
                continue
    return ids


@dataclass
class ApplyStats:
    """What the masked apply did, for logging and for the NaN post-condition."""

    masked_writes: int = 0
    passthrough_writes: int = 0
    unmaskable_writes: int = 0
    """Writes into a parameter that masking could not narrow (shape or dtype
    mismatch). Each one is a chance for NaN to reach the weights."""
    unmaskable_examples: list[str] = field(default_factory=list)


@contextmanager
def masked_copy(*models: torch.nn.Module) -> Iterator[ApplyStats]:
    """Make ``Tensor.copy_`` skip NaN positions when writing model weights.

    Writes to anything that is not backed by one of the models' own parameter or
    buffer storages pass through untouched, so a loader staging through scratch
    memory still sees a faithful copy of the NaN-masked source.

    Pass every model the flush is applied to: a speculative-decoding drafter
    receives the same weights, and one left out of this set would take the
    unmasked path and have the NaNs written into it.
    """
    stats = ApplyStats()
    owned = _storage_ids(models)
    original_copy = torch.Tensor.copy_

    def masked_copy_(self: torch.Tensor, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        try:
            destination_is_weight = self.untyped_storage().data_ptr() in owned
        except Exception:
            destination_is_weight = False

        if not destination_is_weight or not isinstance(src, torch.Tensor):
            stats.passthrough_writes += 1
            return original_copy(self, src, *args, **kwargs)

        if src.is_floating_point() and self.shape == src.shape:
            # Sync-free masked overwrite: boolean advanced indexing (or a
            # ``mask.all()`` early-out) would force a device->host sync per
            # parameter -- ruinous for MoE flushes carrying >10k per-expert
            # entries. ``torch.where`` keeps everything on-stream, and a NaN-free
            # (dense) source degenerates to a plain copy.
            cast = src.to(self.dtype)
            stats.masked_writes += 1
            return original_copy(self, torch.where(torch.isnan(cast), self, cast))

        stats.unmaskable_writes += 1
        if len(stats.unmaskable_examples) < 8:
            stats.unmaskable_examples.append(f"dst{tuple(self.shape)}:{self.dtype} src{tuple(src.shape)}:{src.dtype}")
        return original_copy(self, src, *args, **kwargs)

    torch.Tensor.copy_ = masked_copy_
    try:
        yield stats
    finally:
        torch.Tensor.copy_ = original_copy


def find_nan_parameters(*models: torch.nn.Module) -> list[str]:
    """Names of parameters or buffers left holding NaN.

    Flags are accumulated on the device and read back once, so this costs a
    single synchronisation for the whole model rather than one per parameter.
    Trained weights hold no NaN, so any hit means a write escaped the mask.
    """
    named = []
    for i, model in enumerate(models):
        prefix = "" if len(models) == 1 else f"model{i}."
        named += [(prefix + name, t) for name, t in model.named_parameters() if t.is_floating_point()]
        named += [(prefix + name, t) for name, t in model.named_buffers() if t.is_floating_point()]
    if not named:
        return []

    flags = torch.stack([torch.isnan(t.detach()).any().to(torch.uint8) for _, t in named])
    return [named[i][0] for i in flags.nonzero().flatten().tolist()]


def check_applied(models, stats: ApplyStats, *, strict: bool = True) -> None:
    """Fail loud if the flush left NaN in the weights.

    ``strict=False`` downgrades to a warning for callers that would rather finish
    the sync than abort training on it.
    """
    if isinstance(models, torch.nn.Module):
        models = [models]
    if stats.unmaskable_writes:
        logger.warning(
            "delta apply: %d write(s) into model weights could not be masked (examples: %s); "
            "a NaN-masked source reaching such a write lands verbatim in the weights",
            stats.unmaskable_writes,
            ", ".join(stats.unmaskable_examples),
        )

    nan_params = find_nan_parameters(*models)
    if not nan_params:
        return

    message = (
        f"delta apply left NaN in {len(nan_params)} parameter(s): {nan_params[:12]}"
        f"{' ...' if len(nan_params) > 12 else ''}. The sparse delta marks unchanged positions with "
        f"NaN and relies on the masked copy to skip them, so this means a write bypassed the mask "
        f"({stats.unmaskable_writes} unmaskable write(s) seen: {stats.unmaskable_examples})"
    )
    if strict:
        raise RuntimeError(message)
    logger.error(message)
