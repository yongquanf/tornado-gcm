"""Spatial filters as nn.Module — PyTorch implementation.

Provides ExponentialModalFilter and related spectral-space filters
wrapped as nn.Module for use in neural pipelines.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from pytorch_src.core import filtering
from pytorch_src.core import spherical_harmonic


class ExponentialModalFilter(nn.Module):
    """Exponential modal (spectral) filter as nn.Module.

    s_l = exp(-attenuation * ((l/L - cutoff)/(1 - cutoff))^(2*order))

    This wraps the core filtering.exponential_filter function but stores
    the filter coefficients as a buffer on the module.
    """

    def __init__(
        self,
        grid: spherical_harmonic.Grid,
        attenuation: float = 16.0,
        order: int = 18,
        cutoff: float = 0.0,
    ):
        super().__init__()
        self.grid = grid
        self.attenuation = attenuation
        self.order = order
        self.cutoff = cutoff

        # Precompute filter coefficients
        L = grid.total_wavenumbers
        l_vals = np.arange(L, dtype=np.float64)
        s = np.ones(L, dtype=np.float64)
        mask = l_vals / L > cutoff
        normalized = (l_vals[mask] / L - cutoff) / (1.0 - cutoff)
        s[mask] = np.exp(-attenuation * normalized ** (2 * order))
        self.register_buffer(
            "_coeffs", torch.tensor(s, dtype=torch.float32), persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply exponential filter to modal field."""
        coeffs = self._coeffs.to(dtype=x.dtype, device=x.device)
        return x * coeffs


class HorizontalDiffusionFilter(nn.Module):
    """Horizontal diffusion filter as nn.Module.

    s_l = exp(-tau * (l*(l+1)/a²)^n_diff)
    """

    def __init__(
        self,
        grid: spherical_harmonic.Grid,
        tau: float = 0.01,
        n_diff: int = 1,
    ):
        super().__init__()
        self.grid = grid
        self.tau = tau
        self.n_diff = n_diff

        L = grid.total_wavenumbers
        a = grid.radius
        l_vals = np.arange(L, dtype=np.float64)
        eig = l_vals * (l_vals + 1) / (a**2)
        s = np.exp(-tau * eig**n_diff)
        self.register_buffer(
            "_coeffs", torch.tensor(s, dtype=torch.float32), persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coeffs = self._coeffs.to(dtype=x.dtype, device=x.device)
        return x * coeffs


class ComposedFilter(nn.Module):
    """Compose multiple filters sequentially."""

    def __init__(self, *filters: nn.Module):
        super().__init__()
        self.filters = nn.ModuleList(filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for f in self.filters:
            x = f(x)
        return x
