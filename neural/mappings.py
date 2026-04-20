# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""Neural mappings: NodalMapping and NodalVolumeMapping — PyTorch.

Pack pytree features → NN tower → unpack to target shapes.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn


class NodalMapping(nn.Module):
    """Map nodal features through a tower network.

    Packs input dict fields along the level/feature axis (-3),
    sends through the tower NN, then unpacks to output shapes.

    Args:
        tower: neural network operating on (n_features, lon, lat) → (n_outputs, lon, lat)
        output_keys: ordered list of output variable names.
        output_sizes: number of levels/channels per output key.
    """

    def __init__(
        self,
        tower: nn.Module,
        output_keys: Sequence[str],
        output_sizes: Sequence[int],
    ):
        super().__init__()
        self.tower = tower
        self.output_keys = list(output_keys)
        self.output_sizes = list(output_sizes)

    def _pack(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Pack dict fields along axis -3 into (n_features, lon, lat)."""
        arrays = []
        for key, val in sorted(inputs.items()):
            if val.ndim == 2:
                val = val.unsqueeze(0)
            arrays.append(val)
        return torch.cat(arrays, dim=-3)

    def _unpack(self, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Split tower output into named fields."""
        out = {}
        idx = 0
        for key, sz in zip(self.output_keys, self.output_sizes):
            out[key] = y[..., idx:idx + sz, :, :]
            idx += sz
        return out

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = self._pack(inputs)
        y = self.tower(x)
        return self._unpack(y)


class NodalVolumeMapping(nn.Module):
    """Map nodal volume fields through a tower network, preserving the level axis.

    Input is stacked along channel axis (dim=0), NN processes each level
    independently, then output is unstacked.

    Args:
        tower: NN operating on (n_channels, n_levels, lon, lat) → (n_outputs, n_levels, lon, lat)
        output_keys: ordered list of output variable names.
    """

    def __init__(
        self,
        tower: nn.Module,
        output_keys: Sequence[str],
    ):
        super().__init__()
        self.tower = tower
        self.output_keys = list(output_keys)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Stack all inputs along channel dimension (dim=0)
        arrays = [inputs[k] for k in sorted(inputs.keys())]
        x = torch.stack(arrays, dim=0)  # (n_channels, levels, lon, lat)
        y = self.tower(x)  # (n_outputs, levels, lon, lat)
        out = {}
        for i, key in enumerate(self.output_keys):
            out[key] = y[i]  # (levels, lon, lat)
        return out


class ParallelMapping(nn.Module):
    """Sum of multiple mapping modules applied in parallel.

    result = sum(mapping(inputs) for mapping in mappings)
    """

    def __init__(self, mappings: Sequence[nn.Module]):
        super().__init__()
        self.mappings = nn.ModuleList(mappings)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        results = [m(inputs) for m in self.mappings]
        combined = dict(results[0])
        for r in results[1:]:
            for key in combined:
                if key in r:
                    combined[key] = combined[key] + r[key]
        return combined
