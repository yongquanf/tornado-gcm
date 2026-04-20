# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""Vertical coordinate system for multi-layer shallow water equations."""

from __future__ import annotations

import dataclasses
import numpy as np


@dataclasses.dataclass(frozen=True)
class LayerCoordinates:
    """Vertical coordinate system for multi-layer shallow water equations.

    Layers are indexed from the top of the atmosphere to the surface.

    Attributes:
        layers: the number of layers.
    """

    layers: int

    @property
    def centers(self) -> np.ndarray:
        return np.arange(self.layers)

    def asdict(self):
        return dataclasses.asdict(self)
