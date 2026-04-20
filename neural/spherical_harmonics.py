"""Spherical harmonic mapping wrapper — PyTorch implementation.

Provides FixedYlmMapping that wraps Grid for neural module use.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pytorch_src.core import spherical_harmonic


class FixedYlmMapping(nn.Module):
    """Fixed (non-learnable) spherical harmonic mapping.

    Wraps Grid.to_modal / to_nodal for use in neural pipelines.
    The mapping matrices are registered as buffers (non-trainable).
    """

    def __init__(self, grid: spherical_harmonic.Grid):
        super().__init__()
        self.grid = grid

    def to_nodal(self, x: torch.Tensor) -> torch.Tensor:
        """Modal → nodal transform."""
        return self.grid.to_nodal(x)

    def to_modal(self, x: torch.Tensor) -> torch.Tensor:
        """Nodal → modal transform."""
        return self.grid.to_modal(x)

    def forward(self, x: torch.Tensor, direction: str = "to_nodal") -> torch.Tensor:
        """Apply transform in specified direction."""
        if direction == "to_nodal":
            return self.to_nodal(x)
        elif direction == "to_modal":
            return self.to_modal(x)
        else:
            raise ValueError(f"Unknown direction: {direction}")

    @property
    def nodal_shape(self):
        return self.grid.nodal_shape

    @property
    def modal_shape(self):
        return self.grid.modal_shape


class DivCurlTransform(nn.Module):
    """Compute divergence and curl of a nodal (u, v) field.

    Converts nodal velocity to modal vorticity and divergence.
    """

    def __init__(self, grid: spherical_harmonic.Grid):
        super().__init__()
        self.grid = grid

    def forward(
        self, u_nodal: torch.Tensor, v_nodal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (modal_vorticity, modal_divergence)."""
        return spherical_harmonic.uv_nodal_to_vor_div_modal(
            self.grid, u_nodal, v_nodal,
        )
