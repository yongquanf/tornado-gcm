"""Coordinate system combining horizontal grid and vertical coordinates.

This is the PyTorch port of dinosaur/coordinate_systems.py.
SPMD sharding is omitted (single-GPU MVP).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Sequence, Union

import numpy as np
import torch

from pytorch_src.core import layer_coordinates
from pytorch_src.core import sigma_coordinates
from pytorch_src.core import spherical_harmonic


HorizontalGridTypes = spherical_harmonic.Grid
VerticalCoordinateTypes = Union[
    layer_coordinates.LayerCoordinates,
    sigma_coordinates.SigmaCoordinates,
    Any,
]


@dataclasses.dataclass(frozen=True)
class CoordinateSystem:
    """Combined horizontal and vertical grid data.

    Attributes:
        horizontal: object describing horizontal discretization.
        vertical: object describing vertical discretization.
    """

    horizontal: HorizontalGridTypes
    vertical: VerticalCoordinateTypes

    @property
    def nodal_shape(self) -> tuple[int, int, int]:
        return (self.vertical.layers,) + self.horizontal.nodal_shape

    @property
    def modal_shape(self) -> tuple[int, int, int]:
        return (self.vertical.layers,) + self.horizontal.modal_shape

    @property
    def surface_nodal_shape(self) -> tuple[int, int, int]:
        return (1,) + self.horizontal.nodal_shape

    @property
    def surface_modal_shape(self) -> tuple[int, int, int]:
        return (1,) + self.horizontal.modal_shape


# ═══════════════════════════════════════════════════════════════════════════
# Spectral interpolation helpers
# ═══════════════════════════════════════════════════════════════════════════

def _tree_map_tensors(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: Any,
) -> Any:
    """Apply fn to all Tensor leaves in a pytree-like structure."""
    if isinstance(x, torch.Tensor):
        return fn(x)
    elif isinstance(x, dict):
        return {k: _tree_map_tensors(fn, v) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        result = [_tree_map_tensors(fn, v) for v in x]
        return type(x)(result)
    return x


def get_spectral_downsample_fn(
    coords: CoordinateSystem,
    save_coords: CoordinateSystem,
    expect_same_vertical: bool = True,
) -> Callable:
    """Return fn that downsamples modal state from coords to save_coords."""
    if expect_same_vertical and (coords.vertical != save_coords.vertical):
        raise ValueError("Downsampling vertical resolution is not supported.")
    m_end = save_coords.horizontal.modal_shape[0]
    l_end = save_coords.horizontal.modal_shape[1]

    def downsample(state):
        fn = lambda x: x[..., :m_end, :l_end]
        return _tree_map_tensors(fn, state)
    return downsample


def get_spectral_upsample_fn(
    coords: CoordinateSystem,
    save_coords: CoordinateSystem,
    expect_same_vertical: bool = True,
) -> Callable:
    """Return fn that upsamples modal state from coords to save_coords."""
    if expect_same_vertical and (coords.vertical != save_coords.vertical):
        raise ValueError("Upsampling vertical resolution is not supported.")
    save_shape = save_coords.horizontal.modal_shape
    coords_shape = coords.horizontal.modal_shape
    pad_m = save_shape[0] - coords_shape[0]
    pad_l = save_shape[1] - coords_shape[1]
    if pad_m < 0 or pad_l < 0:
        raise ValueError("save_coords resolution smaller than coords.")

    def upsample(state):
        def pad_fn(x: torch.Tensor) -> torch.Tensor:
            n = x.ndim
            # pad last 2 dims: (left_last, right_last, left_2nd, right_2nd)
            return torch.nn.functional.pad(x, (0, pad_l, 0, pad_m))
        return _tree_map_tensors(pad_fn, state)
    return upsample


def get_spectral_interpolate_fn(
    source_coords: CoordinateSystem,
    target_coords: CoordinateSystem,
    expect_same_vertical: bool = True,
) -> Callable:
    """Return modal interpolation fn from source to target coords."""
    src = source_coords.horizontal
    tgt = target_coords.horizontal
    if src.total_wavenumbers < tgt.total_wavenumbers:
        return get_spectral_upsample_fn(
            source_coords, target_coords, expect_same_vertical
        )
    elif src.total_wavenumbers >= tgt.total_wavenumbers:
        return get_spectral_downsample_fn(
            source_coords, target_coords, expect_same_vertical
        )
    else:
        raise ValueError(
            f"Incompatible horizontal: {src.modal_shape}, {tgt.modal_shape}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Nodal/modal conversion helpers
# ═══════════════════════════════════════════════════════════════════════════

def maybe_to_nodal(
    fields: Any,
    coords: CoordinateSystem,
) -> Any:
    """Convert non-scalar elements to nodal if they are modal."""
    nodal_shape = coords.horizontal.nodal_shape

    def fn(x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == nodal_shape:
            return x
        return coords.horizontal.to_nodal(x)

    return _tree_map_tensors(fn, fields)


def maybe_to_modal(
    fields: Any,
    coords: CoordinateSystem,
) -> Any:
    """Convert non-scalar elements to modal if they are nodal."""
    modal_shape = coords.horizontal.modal_shape

    def fn(x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == modal_shape:
            return x
        return coords.horizontal.to_modal(x)

    return _tree_map_tensors(fn, fields)


def scale_levels_for_matching_keys(
    inputs: dict[str, Any],
    scales: torch.Tensor,
    keys_to_scale: Sequence[str] = (),
) -> dict[str, Any]:
    """Scale selected keys along the vertical (level) axis."""
    if scales.ndim != 1:
        raise ValueError(
            f"scales must be 1d, got shape {scales.shape}"
        )
    s = scales[:, None, None]  # (levels, 1, 1) for broadcasting

    def _recurse(d: dict) -> dict:
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out[k] = _recurse(v)
            elif k in keys_to_scale and isinstance(v, torch.Tensor):
                out[k] = v * s
            else:
                out[k] = v
        return out

    return _recurse(inputs)
