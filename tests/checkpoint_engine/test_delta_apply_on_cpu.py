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

"""The masked apply has to hold up against the shapes real weight loaders take.

Each test stands in for one thing an engine's ``load_weights`` is known to do --
write a narrowed slice, stage through scratch memory, reshape, derive values
arithmetically -- because the delta's "NaN means unchanged" convention is only
safe as long as every one of those either gets masked or gets caught.
"""

import pytest
import torch

from verl.checkpoint_engine.delta_sync.apply import (
    check_applied,
    decode_param,
    find_nan_parameters,
    masked_copy,
)


class Tiny(torch.nn.Module):
    def __init__(self, shape=(4, 3)):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(*shape))


def sparse(shape, dtype, changes: dict[int, float]):
    """Build the (positions, values, manifest) triple the wire carries."""
    idx = sorted(changes)
    positions = torch.tensor(idx, dtype=torch.int32).view(torch.uint8)
    values = torch.tensor([changes[i] for i in idx], dtype=dtype)
    param = {
        "name": "weight",
        "dtype": str(dtype).removeprefix("torch."),
        "shape": list(shape),
        "pos_start": 0,
        "pos_end": positions.numel(),
        "pos_width": 4,
        "val_start": 0,
        "val_end": values.numel(),
    }
    return positions, values, param


def test_decode_marks_untouched_positions_with_nan():
    positions, values, param = sparse((2, 3), torch.float32, {1: 5.0, 4: -2.0})

    dense = decode_param("indices", positions, values, param)

    assert dense.shape == (2, 3)
    assert dense[0, 1] == 5.0
    assert dense[1, 1] == -2.0
    assert torch.isnan(dense).sum() == 4


def test_decode_rejects_unknown_encoding():
    positions, values, param = sparse((2, 3), torch.float32, {0: 1.0})

    with pytest.raises(ValueError, match="unsupported delta encoding"):
        decode_param("bitmap", positions, values, param)


def test_masked_copy_writes_only_changed_positions():
    model = Tiny((2, 3))
    model.weight.data.fill_(7.0)
    positions, values, param = sparse((2, 3), torch.float32, {1: 5.0})
    dense = decode_param("indices", positions, values, param)

    with masked_copy(model) as stats:
        model.weight.data.copy_(dense)

    assert model.weight[0, 1] == 5.0
    assert model.weight[0, 0] == 7.0, "an unchanged position must keep its old value"
    assert stats.masked_writes == 1
    assert find_nan_parameters(model) == []


def test_masked_copy_handles_a_narrowed_destination():
    """Loaders write fused projections one shard at a time, so the destination is
    a view into the parameter rather than the parameter itself."""
    model = Tiny((4, 3))
    model.weight.data.fill_(7.0)
    dense = torch.full((2, 3), float("nan"))
    dense[1, 2] = 1.0

    with masked_copy(model) as stats:
        model.weight.data.narrow(0, 2, 2).copy_(dense)

    assert model.weight[3, 2] == 1.0
    assert model.weight[2, 0] == 7.0
    assert stats.masked_writes == 1


def test_scratch_tensor_receives_the_nans_verbatim():
    """A loader that stages through scratch memory must see a faithful copy.

    Masking a scratch destination would replace the NaNs with whatever that
    buffer held; the values would no longer be NaN, and the next write into the
    parameter would have nothing left to mask.
    """
    model = Tiny((2, 3))
    model.weight.data.fill_(7.0)
    dense = torch.full((2, 3), float("nan"))
    dense[0, 1] = 5.0

    with masked_copy(model) as stats:
        scratch = torch.full((2, 3), -999.0)
        scratch.copy_(dense)
        model.weight.data.copy_(scratch)

    assert torch.isnan(scratch).sum() == 5, "scratch must keep the NaNs, not the -999 it held"
    assert model.weight[0, 1] == 5.0
    assert model.weight[0, 0] == 7.0, "staging through scratch must not corrupt unchanged weights"
    assert stats.passthrough_writes == 1
    assert stats.masked_writes == 1


def test_broadcast_write_cannot_be_masked_and_the_sweep_catches_it():
    """``copy_`` broadcasts, so a loader may hand over a source of a different
    shape. Masking needs matching shapes, so such a write falls through and the
    NaNs land in the parameter -- which is what the end-of-flush sweep is for."""
    model = Tiny((2, 3))
    model.weight.data.fill_(7.0)
    row = torch.full((1, 3), float("nan"))
    row[0, 1] = 5.0

    with masked_copy(model) as stats:
        model.weight.data.copy_(row)

    assert stats.unmaskable_writes == 1
    assert stats.masked_writes == 0
    assert find_nan_parameters(model) == ["weight"], "the NaNs did reach the weights"
    with pytest.raises(RuntimeError, match="left NaN"):
        check_applied(model, stats)


def test_without_the_ownership_guard_scratch_staging_corrupts_weights():
    """Why the destination has to be checked at all.

    This reproduces the pre-guard behaviour -- mask every floating-point write of
    a matching shape, whatever it is writing to -- and shows it silently replaces
    an unchanged weight with the scratch buffer's contents. The guarded version of
    the same sequence is asserted in
    ``test_scratch_tensor_receives_the_nans_verbatim``.
    """
    model = Tiny((2, 3))
    model.weight.data.fill_(7.0)
    dense = torch.full((2, 3), float("nan"))
    dense[0, 1] = 5.0

    original_copy = torch.Tensor.copy_

    def mask_every_destination(self, src, *args, **kwargs):
        if isinstance(src, torch.Tensor) and src.is_floating_point() and self.shape == src.shape:
            cast = src.to(self.dtype)
            return original_copy(self, torch.where(torch.isnan(cast), self, cast))
        return original_copy(self, src, *args, **kwargs)

    torch.Tensor.copy_ = mask_every_destination
    try:
        scratch = torch.full((2, 3), -999.0)
        scratch.copy_(dense)
        model.weight.data.copy_(scratch)
    finally:
        torch.Tensor.copy_ = original_copy

    assert model.weight[0, 1] == 5.0, "the changed position still arrives"
    assert model.weight[0, 0] == -999.0, (
        "an unchanged weight was overwritten with the scratch buffer's contents, "
        "and no NaN is left for a sweep to detect"
    )


def test_integer_destination_falls_through_and_is_flagged():
    """Quantised parameters are not floating point, so nothing can be masked."""

    class Quantised(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("qweight", torch.zeros(4, dtype=torch.int8))

    model = Quantised()
    with masked_copy(model) as stats:
        model.qweight.copy_(torch.tensor([1, 2, 3, 4], dtype=torch.int8))

    assert stats.unmaskable_writes == 1
    assert stats.unmaskable_examples


def test_check_applied_raises_on_nan_left_behind():
    model = Tiny((2, 3))
    with masked_copy(model) as stats:
        pass
    model.weight.data[0, 0] = float("nan")

    with pytest.raises(RuntimeError, match="left NaN in 1 parameter"):
        check_applied(model, stats)


def test_check_applied_can_warn_instead_of_raising():
    model = Tiny((2, 3))
    with masked_copy(model) as stats:
        pass
    model.weight.data[0, 0] = float("nan")

    check_applied(model, stats, strict=False)


def test_copy_is_restored_even_when_the_body_raises():
    model = Tiny()
    original = torch.Tensor.copy_

    with pytest.raises(RuntimeError, match="loader blew up"):
        with masked_copy(model):
            raise RuntimeError("loader blew up")

    assert torch.Tensor.copy_ is original


def test_dense_source_is_written_unchanged():
    """The first flush of a stream carries full values, not a sparse delta; it
    must not be slowed down or altered by the masking path."""
    model = Tiny((2, 3))
    model.weight.data.fill_(7.0)
    dense = torch.arange(6, dtype=torch.float32).view(2, 3)

    with masked_copy(model):
        model.weight.data.copy_(dense)

    assert torch.equal(model.weight.detach(), dense)


def test_find_nan_parameters_reports_names():
    model = Tiny((2, 3))
    assert find_nan_parameters(model) == []

    model.weight.data[1, 1] = float("nan")
    assert find_nan_parameters(model) == ["weight"]
