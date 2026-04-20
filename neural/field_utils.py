# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""Field utility functions — PyTorch implementation.

Provides utilities for reconstructing, slicing, and manipulating
fields in both nodal and modal representations.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def reconstruct_1d_field_from_ref_values(
    field: torch.Tensor,
    ref_values: torch.Tensor,
    axis: int = -3,
) -> torch.Tensor:
    """Add reference profile back to a perturbation field.

    Args:
        field: perturbation field (e.g. T').
        ref_values: 1D reference profile (e.g. T_ref at each level).
        axis: vertical axis.

    Returns:
        field + ref_values (broadcast along axis).
    """
    shape = [1] * field.ndim
    pos = axis if axis >= 0 else field.ndim + axis
    shape[pos] = -1
    ref = ref_values.reshape(shape).to(field.dtype).to(field.device)
    return field + ref


def extract_1d_field_perturbation(
    field: torch.Tensor,
    ref_values: torch.Tensor,
    axis: int = -3,
) -> torch.Tensor:
    """Subtract reference profile from a full field.

    Args:
        field: full field (e.g. T).
        ref_values: 1D reference profile.
        axis: vertical axis.

    Returns:
        field - ref_values.
    """
    shape = [1] * field.ndim
    pos = axis if axis >= 0 else field.ndim + axis
    shape[pos] = -1
    ref = ref_values.reshape(shape).to(field.dtype).to(field.device)
    return field - ref


def slice_along_axis(
    x: torch.Tensor,
    start: int,
    length: int,
    axis: int = -3,
) -> torch.Tensor:
    """Slice a tensor along the given axis."""
    pos = axis if axis >= 0 else x.ndim + axis
    return x.narrow(pos, start, length)


def pad_along_axis(
    x: torch.Tensor,
    pad_before: int,
    pad_after: int,
    axis: int = -3,
    value: float = 0.0,
) -> torch.Tensor:
    """Pad a tensor along the given axis."""
    pos = axis if axis >= 0 else x.ndim + axis
    n_dims = x.ndim
    # torch.nn.functional.pad expects (last_dim_pad, ..., first_dim_pad)
    pad_list = [0] * (2 * n_dims)
    idx = 2 * (n_dims - 1 - pos)
    pad_list[idx] = pad_before
    pad_list[idx + 1] = pad_after
    return torch.nn.functional.pad(x, pad_list, value=value)


def weighted_vertical_mean(
    x: torch.Tensor,
    weights: torch.Tensor,
    axis: int = -3,
    keepdim: bool = True,
) -> torch.Tensor:
    """Compute weighted mean along vertical axis."""
    pos = axis if axis >= 0 else x.ndim + axis
    shape = [1] * x.ndim
    shape[pos] = -1
    w = weights.reshape(shape).to(x.dtype).to(x.device)
    return (x * w).sum(dim=pos, keepdim=keepdim) / w.sum()


def repeat_surface_to_volume(
    surface: torch.Tensor,
    n_layers: int,
    axis: int = -3,
) -> torch.Tensor:
    """Repeat a surface field (2D) to fill all vertical levels."""
    pos = axis if axis >= 0 else surface.ndim + 1 + axis
    return surface.unsqueeze(pos).expand(
        *surface.shape[:pos], n_layers, *surface.shape[pos:]
    )


def vertical_flip(x: torch.Tensor, axis: int = -3) -> torch.Tensor:
    """Flip tensor along vertical axis (top↔surface)."""
    pos = axis if axis >= 0 else x.ndim + axis
    return torch.flip(x, dims=[pos])
