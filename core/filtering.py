"""Spectral filtering — exponential and horizontal diffusion filters.

All filters operate in modal (spectral) space by computing per-wavenumber
scaling coefficients and applying them element-wise.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from pytorch_src.core.spherical_harmonic import Grid


def exponential_filter(
    grid: Grid,
    attenuation: float = 16.0,
    order: int = 18,
    cutoff: float = 0.0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build an exponential spectral filter.

    Args:
        grid: the spherical harmonic grid.
        attenuation: -log of the filter value at the highest wavenumber.
        order: exponent order of the filter.
        cutoff: fraction of wavenumbers below which no filtering is applied.

    Returns:
        A function that applies the filter to a modal tensor.
    """
    _, k = grid.modal_axes
    k_max = k[-1]
    if k_max == 0:
        def identity(x: torch.Tensor) -> torch.Tensor:
            return x
        return identity
    k_frac = k / k_max
    c = cutoff
    with np.errstate(divide="ignore", invalid="ignore"):
        arg = ((k_frac - c) / (1 - c)) ** (2 * order)
    arg = np.where(k_frac > c, arg, 0.0)
    scaling = np.exp(-attenuation * arg)

    def apply_filter(x: torch.Tensor) -> torch.Tensor:
        s = torch.tensor(scaling, dtype=x.dtype, device=x.device)
        return x * s
    return apply_filter


def horizontal_diffusion_filter(
    grid: Grid,
    scale: float,
    order: int = 1,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a horizontal diffusion filter.

    Applies exp(scale * (-eigenvalues)^order) as scaling.

    Args:
        grid: the spherical harmonic grid.
        scale: diffusion coefficient.
        order: order of the diffusion operator.

    Returns:
        A function that applies diffusion filtering to a modal tensor.
    """
    eigenvalues = grid.laplacian_eigenvalues
    scaling = np.exp(scale * (-eigenvalues) ** order)

    def apply_filter(x: torch.Tensor) -> torch.Tensor:
        s = torch.tensor(scaling, dtype=x.dtype, device=x.device)
        return x * s
    return apply_filter
