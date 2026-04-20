"""Neural-level coordinate containers — PyTorch implementation.

Lightweight wrappers around core coordinate objects that provide
the interface expected by neural modules.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
import torch

from pytorch_src.core import sigma_coordinates as sigma_mod
from pytorch_src.core import hybrid_coordinates as hybrid_mod


@dataclasses.dataclass(frozen=True)
class TimeDelta:
    """Nondimensional time step container."""
    value: float

    def __float__(self) -> float:
        return self.value


@dataclasses.dataclass(frozen=True)
class SigmaLevels:
    """Sigma-level vertical coordinates for neural modules."""
    sigma: sigma_mod.SigmaCoordinates

    @property
    def centers(self) -> np.ndarray:
        return self.sigma.centers

    @property
    def boundaries(self) -> np.ndarray:
        return self.sigma.boundaries

    @property
    def layers(self) -> int:
        return self.sigma.layers

    @property
    def layer_thickness(self) -> np.ndarray:
        return self.sigma.layer_thickness

    @classmethod
    def equidistant(cls, n_layers: int) -> SigmaLevels:
        return cls(sigma_mod.SigmaCoordinates.equidistant(n_layers))


@dataclasses.dataclass(frozen=True)
class HybridLevels:
    """Hybrid sigma-pressure levels for neural modules."""
    hybrid: hybrid_mod.HybridCoordinates

    @property
    def layers(self) -> int:
        return self.hybrid.layers

    @property
    def a_boundaries(self) -> np.ndarray:
        return self.hybrid.a_boundaries

    @property
    def b_boundaries(self) -> np.ndarray:
        return self.hybrid.b_boundaries

    def pressure_at_level(self, surface_pressure: torch.Tensor) -> torch.Tensor:
        return self.hybrid.pressure_centers(surface_pressure)

    @classmethod
    def from_sigma(cls, sigma: sigma_mod.SigmaCoordinates) -> HybridLevels:
        return cls(hybrid_mod.HybridCoordinates.from_sigma_levels(sigma))


@dataclasses.dataclass(frozen=True)
class PressureLevels:
    """Fixed pressure levels for observations / output."""
    levels: np.ndarray  # pressure values (Pa or hPa)

    @property
    def n_levels(self) -> int:
        return len(self.levels)

    @classmethod
    def era5_37(cls) -> PressureLevels:
        """Standard ERA5 37-level pressure grid."""
        return cls(np.array([
            1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200,
            225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750,
            775, 800, 825, 850, 875, 900, 925, 950, 975, 1000,
        ], dtype=np.float64))

    @classmethod
    def from_list(cls, levels: list[float]) -> PressureLevels:
        return cls(np.array(levels, dtype=np.float64))
