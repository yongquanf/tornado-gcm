# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""Spectral filtering — exponential and horizontal diffusion filters.

All filters operate in modal (spectral) space by computing per-wavenumber
scaling coefficients and applying them element-wise.

Filters are ``nn.Module`` subclasses so that ``model.to(device)`` moves
their scaling tensors to the GPU automatically, avoiding CUDA-graph
DeviceCopy partitions.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tornado_gcm.core.spherical_harmonic import Grid


class ExponentialFilter(nn.Module):
    """Exponential spectral filter as an nn.Module.

    scaling[k] = exp(-attenuation * ((k_frac-cutoff)/(1-cutoff))^(2*order))
    for k_frac > cutoff; 1.0 otherwise.
    """

    def __init__(self, grid: Grid, attenuation: float = 16.0,
                 order: int = 18, cutoff: float = 0.0):
        super().__init__()
        _, k_np = grid.modal_axes
        k = torch.from_numpy(k_np.copy()).to(torch.float64)
        k_max = k[-1].item()
        if k_max == 0:
            self.register_buffer("scaling", torch.ones(1))
            return
        k_frac = k / k_max
        c = cutoff
        arg = ((k_frac - c) / (1 - c)) ** (2 * order)
        arg = torch.where(k_frac > c, arg, torch.zeros_like(arg))
        scaling = torch.exp(-attenuation * arg).to(torch.float32)
        self.register_buffer("scaling", scaling, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # .to(device=…) guarded by is_compiling() so Dynamo dead-code-eliminates
        # it from the FX graph (no CUDA-graph DeviceCopy partition).
        # The scaling buffer follows model.to(device) so it is always on the
        # correct device at compile time; the guard handles pre-compile edge cases.
        s = self.scaling
        if not torch.compiler.is_compiling() and s.device != x.device:
            s = s.to(device=x.device)
        return x * s.to(dtype=x.dtype)


class HorizontalDiffusionFilter(nn.Module):
    """Horizontal diffusion filter as an nn.Module.

    scaling[k] = exp(scale * (-eigenvalues)^order)
    """

    def __init__(self, grid: Grid, scale: float, order: int = 1):
        super().__init__()
        eigenvalues_np = grid.laplacian_eigenvalues
        eigenvalues = torch.from_numpy(eigenvalues_np.copy()).to(torch.float64)
        scaling = torch.exp(scale * (-eigenvalues) ** order).to(torch.float32)
        self.register_buffer("scaling", scaling, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # .to(device=…) guarded by is_compiling() so Dynamo dead-code-eliminates
        # it from the FX graph (no CUDA-graph DeviceCopy partition).
        # The scaling buffer follows model.to(device) so it is always on the
        # correct device at compile time; the guard handles pre-compile edge cases.
        s = self.scaling
        if not torch.compiler.is_compiling() and s.device != x.device:
            s = s.to(device=x.device)
        return x * s.to(dtype=x.dtype)


class CombinedExponentialFilter(nn.Module):
    """Product of two spectral filters applied in a single pass.

    scaling = scaling_a * scaling_b  (pre-computed at init time).
    """

    def __init__(self, scaling: torch.Tensor):
        super().__init__()
        self.register_buffer(
            "scaling",
            scaling.to(torch.float32),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.scaling
        if not torch.compiler.is_compiling() and s.device != x.device:
            s = s.to(device=x.device)
        return x * s.to(dtype=x.dtype)


# Backward-compatible factory functions
def exponential_filter(
    grid: Grid,
    attenuation: float = 16.0,
    order: int = 18,
    cutoff: float = 0.0,
) -> ExponentialFilter:
    return ExponentialFilter(grid, attenuation, order, cutoff)


def horizontal_diffusion_filter(
    grid: Grid,
    scale: float,
    order: int = 1,
) -> HorizontalDiffusionFilter:
    return HorizontalDiffusionFilter(grid, scale, order)
