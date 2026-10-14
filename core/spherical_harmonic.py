# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""Spherical harmonics basis and differential operators — PyTorch implementation.

Zones:
    Z0 — Basis matrices (Legendre, Fourier) computed and stored in float64.
    Z1 — Transform einsums use configurable precision (einsum_highest or TF32).

This module implements RealSphericalHarmonics for correctness and the Grid class.
FastSphericalHarmonics is not ported (no SPMD mesh support in single-card MVP).
"""

from __future__ import annotations

import dataclasses
import functools
import math
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from tornado_gcm.core import associated_legendre
from tornado_gcm.core import fourier
from tornado_gcm.precision.zone_cast import einsum_highest


# ═══════════════════════════════════════════════════════════════════════════
# Adaptive precision einsum (Algorithm 8)
# ═══════════════════════════════════════════════════════════════════════════

def einsum_adaptive(subscripts: str, *operands: torch.Tensor) -> torch.Tensor:
    """Einsum in FP32 with TF32 matmul (lower precision than einsum_highest).

    Used for lower-resolution grids (L≤42) where FP64 intermediates are
    unnecessary. Relies on torch.backends.cuda.matmul.allow_tf32 being set.
    """
    ops32 = [op.to(torch.float32) for op in operands]
    result = torch.einsum(subscripts, *ops32)
    return result.to(torch.float32)


def get_sht_einsum(total_wavenumbers: int, threshold: int = 42):
    """Select the SHT einsum function based on resolution (Algorithm 8).

    - L ≤ threshold: FP32 einsum (TF32 on GPU) — faster, sufficient precision.
    - L > threshold: FP64 einsum (einsum_highest) — needed for numerical stability.

    This implements the adaptive precision strategy from 混合精度加速完整方案.tex
    Algorithm 8: at low resolution, SHT intermediates can safely use lower
    precision without impacting forecast quality.

    Args:
        total_wavenumbers: maximum total wavenumber L of the spectral grid.
        threshold: wavenumber threshold below which FP32/TF32 is used.
            Default 42. Can be overridden via PrecisionPolicy.sht_adaptive_threshold.
    """
    if not isinstance(threshold, int):
        raise ValueError(
            "sht threshold must be an int, got %r; refusing to guess"
            % (type(threshold).__name__,)
        )
    if total_wavenumbers <= threshold:
        return einsum_adaptive
    return einsum_highest


# ═══════════════════════════════════════════════════════════════════════════
# Latitude spacing registry
# ═══════════════════════════════════════════════════════════════════════════

LATITUDE_SPACINGS = dict(
    gauss=associated_legendre.gauss_legendre_nodes,
    equiangular=associated_legendre.equiangular_nodes,
    equiangular_with_poles=associated_legendre.equiangular_nodes_with_poles,
)


def get_latitude_nodes(n: int, spacing: str) -> tuple[np.ndarray, np.ndarray]:
    get_nodes = LATITUDE_SPACINGS.get(spacing)
    if get_nodes is None:
        raise ValueError(
            f"Unknown spacing: {spacing}. "
            f"Available: {list(LATITUDE_SPACINGS.keys())}"
        )
    return get_nodes(n)


# ═══════════════════════════════════════════════════════════════════════════
# Spherical harmonic basis data
# ═══════════════════════════════════════════════════════════════════════════

class _SphericalHarmonicBasis(nn.Module):
    """Precomputed basis matrices for spherical harmonic transforms.

    Inherits from ``nn.Module`` so Dynamo natively traces p/f/w as buffers
    with correct FakeTensor device propagation — no graph breaks, no
    pytree registration needed, no ``@torch.compiler.disable`` needed.

    Buffers:
        p: Legendre basis  (m, j, l) float64
        f: Fourier basis   (i, m)    float64
        w: quadrature weights (j,)   float64
    """

    def __init__(self, p: torch.Tensor, f: torch.Tensor, w: torch.Tensor):
        super().__init__()
        self.register_buffer("p", p)
        self.register_buffer("f", f)
        self.register_buffer("w", w)


# Import here (after _SphericalHarmonicBasis is defined) for the pytree
# registrations below (RealSphericalHarmonics, Grid).
from torch.utils._pytree import register_pytree_node as _reg_pytree


class _GridBuffers(nn.Module):
    """Internal nn.Module holding Grid's static tensors as buffers.

    Registered as ``grid._buf`` inside each Grid.  Dynamo natively traces
    buffer access — no ``getattr(UserDefinedObjectVariable, ...)`` issues,
    no graph breaks, no ``@torch.compiler.disable`` needed.
    """

    def __init__(self):
        super().__init__()
        self._finalized = False

    def _finalize(self):
        """Register all transient attrs as persistent buffers."""
        for _a in (
            "_cos_lat_torch", "_sec2_lat_torch", "_laplacian_eigenvalues_torch",
            "_deriv_a_torch", "_deriv_b_torch", "_modal_l_torch",
            "_lon_torch", "_sin_lat_torch",
        ):
            _init_name = "_init_" + _a[1:]  # _a = "_cos_lat_torch" → "_init_cos_lat_torch"
            _t = getattr(self, _init_name)
            delattr(self, _init_name)
            self.register_buffer(_a, _t)
        self._finalized = True

# ═══════════════════════════════════════════════════════════════════════════
# RealSphericalHarmonics
# ═══════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass(frozen=True)
class RealSphericalHarmonics:
    """Real spherical harmonics implementation (pedagogical, non-parallel).

    Modal representation shape: (2*M-1, L) where
        m = [0, +1, -1, +2, -2, ..., +M, -M]
        l = [0, 1, 2, ..., L-1]
    Entries with |m| > l are structural zeros.
    """

    longitude_wavenumbers: int = 0
    total_wavenumbers: int = 0
    longitude_nodes: int = 0
    latitude_nodes: int = 0
    latitude_spacing: str = "gauss"
    einsum_fn: Any = None  # Adaptive einsum function (Algorithm 8)
    _device: torch.device = torch.device('cpu')

    def __post_init__(self):
        # Pre-compute all lazy values eagerly so torch.compile (Dynamo)
        # never sees a functools.cached_property write on a frozen
        # dataclass — older Dynamo versions reject that pattern.
        _einsum = (self.einsum_fn if self.einsum_fn is not None
                   else get_sht_einsum(self.total_wavenumbers))
        object.__setattr__(self, "_einsum_fn", _einsum)

        longitude, _ = fourier.quadrature_nodes(self.longitude_nodes)
        sin_latitude, _ = get_latitude_nodes(
            self.latitude_nodes, self.latitude_spacing)
        object.__setattr__(self, "_nodal_axes", (longitude, sin_latitude))
        object.__setattr__(self, "_nodal_shape",
                           (self.longitude_nodes, self.latitude_nodes))
        object.__setattr__(self, "_nodal_padding", (0, 0))

        m_pos = np.arange(1, self.longitude_wavenumbers)
        m_pos_neg = np.stack([m_pos, -m_pos], axis=1).ravel()
        lon_wavenumbers = np.concatenate([[0], m_pos_neg])
        tot_wavenumbers = np.arange(self.total_wavenumbers)
        object.__setattr__(self, "_modal_axes",
                           (lon_wavenumbers, tot_wavenumbers))
        object.__setattr__(self, "_modal_shape",
                           (2 * self.longitude_wavenumbers - 1,
                            self.total_wavenumbers))
        object.__setattr__(self, "_modal_padding", (0, 0))
        object.__setattr__(self, "_modal_dtype", np.dtype(np.float32))
        m, l = np.meshgrid(lon_wavenumbers, tot_wavenumbers, indexing="ij")
        object.__setattr__(self, "_mask", abs(m) <= l)

        f = fourier.real_basis(
            wavenumbers=self.longitude_wavenumbers,
            nodes=self.longitude_nodes)
        _, wf = fourier.quadrature_nodes(self.longitude_nodes)
        x, wp = get_latitude_nodes(self.latitude_nodes, self.latitude_spacing)
        w = wf * wp
        p = associated_legendre.evaluate(
            n_m=self.longitude_wavenumbers,
            n_l=self.total_wavenumbers, x=x)
        p = np.repeat(p, 2, axis=0)
        p = p[1:]
        _dev = self._device
        object.__setattr__(self, "_basis",
                           _SphericalHarmonicBasis(
                               f=torch.from_numpy(f.copy()).to(device=_dev, dtype=torch.float64),
                               p=torch.from_numpy(p.copy()).to(device=_dev, dtype=torch.float64),
                               w=torch.from_numpy(w.copy()).to(device=_dev, dtype=torch.float32),
                           ))

    @property
    def _einsum(self):
        return self._einsum_fn

    @property
    def nodal_axes(self) -> tuple[np.ndarray, np.ndarray]:
        return self._nodal_axes

    @property
    def nodal_shape(self) -> tuple[int, int]:
        return self._nodal_shape

    @property
    def nodal_padding(self) -> tuple[int, int]:
        return self._nodal_padding

    @property
    def modal_axes(self) -> tuple[np.ndarray, np.ndarray]:
        return self._modal_axes

    @property
    def modal_shape(self) -> tuple[int, int]:
        return self._modal_shape

    @property
    def modal_padding(self) -> tuple[int, int]:
        return self._modal_padding

    @property
    def modal_dtype(self) -> np.dtype:
        return self._modal_dtype

    @property
    def mask(self) -> np.ndarray:
        return self._mask

    @property
    def basis(self) -> _SphericalHarmonicBasis:
        return self._basis

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Modal → nodal transform.

        Respects ``_subzone_dtype_override`` if set on the Grid — allows
        DAG-discovered sub-zone precision (e.g., Z1_SHT=FP64) to override
        the input tensor's dtype for the transform basis.
        """
        _einsum = self._einsum
        _b = self.basis
        _dtype = getattr(self, "_subzone_dtype_override", None) or x.dtype
        p = _b.p.to(device=x.device, dtype=_dtype)
        f = _b.f.to(device=x.device, dtype=_dtype)
        # Legendre: mjl,...ml -> ...mj
        px = _einsum("mjl,...ml->...mj", p, x)
        # Fourier: im,...mj -> ...ij
        fpx = _einsum("im,...mj->...ij", f, px)
        return fpx

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Nodal → modal transform.

        Respects ``_subzone_dtype_override`` if set on the Grid (symmetric with
        ``inverse_transform``) — so DAG-discovered Z1_SHT precision controls the
        forward transform too, giving 2-way SHT precision control.
        """
        _einsum = self._einsum
        _b = self.basis
        _dtype = getattr(self, "_subzone_dtype_override", None) or x.dtype
        w = _b.w.to(device=x.device, dtype=_dtype)
        p = _b.p.to(device=x.device, dtype=_dtype)
        f = _b.f.to(device=x.device, dtype=_dtype)
        try:
            wx = w * x
        except RuntimeError as e:
            raise RuntimeError(
                f"SHT transform shape mismatch: w.shape={tuple(w.shape)}, "
                f"x.shape={tuple(x.shape)}, x.dtype={x.dtype}, "
                f"nodal_shape={self.nodal_shape}, modal_shape={self.modal_shape}"
            ) from e
        # Fourier: im,...ij -> ...mj
        fwx = _einsum("im,...ij->...mj", f, wx)
        # Legendre: mjl,...mj -> ...ml
        pfwx = _einsum("mjl,...mj->...ml", p, fwx)
        return pfwx

    def longitudinal_derivative(self, x: torch.Tensor) -> torch.Tensor:
        """∂x/∂λ in modal basis."""
        return fourier.real_basis_derivative(x, axis=-2)


# RealSphericalHarmonics holds the basis as a pytree-registered child so
# Dynamo properly traces basis.p / basis.f / basis.w inside inverse_transform
# and transform when called through Grid.to_nodal / Grid.to_modal.
def __rsh_flatten(rsh: RealSphericalHarmonics):
    basis = rsh.basis
    children = (basis,)
    context = {
        "longitude_wavenumbers": rsh.longitude_wavenumbers,
        "total_wavenumbers": rsh.total_wavenumbers,
        "longitude_nodes": rsh.longitude_nodes,
        "latitude_nodes": rsh.latitude_nodes,
        "latitude_spacing": rsh.latitude_spacing,
        "einsum_fn": rsh.einsum_fn,
        "_device": rsh._device,
    }
    return children, context


def __rsh_unflatten(children, context):
    (basis,) = children
    rsh = RealSphericalHarmonics(
        longitude_wavenumbers=context["longitude_wavenumbers"],
        total_wavenumbers=context["total_wavenumbers"],
        longitude_nodes=context["longitude_nodes"],
        latitude_nodes=context["latitude_nodes"],
        latitude_spacing=context["latitude_spacing"],
        einsum_fn=context["einsum_fn"],
        _device=context["_device"],
    )
    object.__setattr__(rsh, "_basis", basis)
    return rsh


_reg_pytree(RealSphericalHarmonics, __rsh_flatten, __rsh_unflatten)


# ═══════════════════════════════════════════════════════════════════════════
# Grid — the main user-facing class
# ═══════════════════════════════════════════════════════════════════════════

SphericalHarmonicsImpl = Callable[..., RealSphericalHarmonics]

_CONSTANT_NORMALIZATION_FACTOR = 3.5449077


@dataclasses.dataclass
class Grid(nn.Module):
    """Representation of real-space and spectral grids over the sphere.

    Attributes:
        longitude_wavenumbers: max (exclusive) longitudinal wavenumber.
        total_wavenumbers: max (exclusive) total wavenumber.
        longitude_nodes: number of longitudinal nodes.
        latitude_nodes: number of latitudinal nodes.
        latitude_spacing: 'gauss', 'equiangular', or 'equiangular_with_poles'.
        longitude_offset: offset for first longitude node (radians).
        radius: sphere radius (default 1.0).
        spherical_harmonics_impl: implementation class for SH transforms.
    """

    longitude_wavenumbers: int = 0
    total_wavenumbers: int = 0
    longitude_nodes: int = 0
    latitude_nodes: int = 0
    latitude_spacing: str = "gauss"
    longitude_offset: float = 0.0
    radius: float = 1.0
    spherical_harmonics_impl: SphericalHarmonicsImpl = RealSphericalHarmonics
    _initial_device: torch.device = torch.device('cpu')

    def __post_init__(self):
        super().__init__()
        if self.radius is None:
            object.__setattr__(self, "radius", 1.0)
        if self.latitude_spacing not in LATITUDE_SPACINGS:
            raise ValueError(
                f'Unsupported latitude_spacing "{self.latitude_spacing}". '
                f"Supported: {list(LATITUDE_SPACINGS)}."
            )
        # Pre-compute ALL lazy values eagerly as plain attributes so that
        # torch.compile (Dynamo) never sees a functools.cached_property
        # write on a frozen dataclass — older Dynamo versions reject that
        # pattern with FrozenDataClassVariable.
        _dev = self._initial_device
        # Effective einsum, in priority order: (1) a fn bound by with_einsum()
        # (stored as a partial keyword), (2) the resolution default.  Passing
        # the call-site kwarg unconditionally used to override (1), so
        # with_einsum() silently had no effect on the transform -- a
        # declared-but-not-executed precision.
        _bound = getattr(self.spherical_harmonics_impl, "keywords", None) or {}
        _eff_einsum = _bound.get("einsum_fn")
        if _eff_einsum is None:
            _eff_einsum = get_sht_einsum(self.total_wavenumbers)
        sh = self.spherical_harmonics_impl(
            longitude_wavenumbers=self.longitude_wavenumbers,
            total_wavenumbers=self.total_wavenumbers,
            longitude_nodes=self.longitude_nodes,
            latitude_nodes=self.latitude_nodes,
            latitude_spacing=self.latitude_spacing,
            einsum_fn=_eff_einsum,
            _device=_dev,
        )
        object.__setattr__(self, "_spherical_harmonics", sh)

        lon, sin_lat = sh.nodal_axes
        _nodal_axes = (lon + self.longitude_offset, sin_lat)
        object.__setattr__(self, "_nodal_axes", _nodal_axes)
        object.__setattr__(self, "_nodal_shape", sh.nodal_shape)
        object.__setattr__(self, "_nodal_padding", sh.nodal_padding)
        object.__setattr__(self, "_nodal_mesh",
                           np.meshgrid(*_nodal_axes, indexing="ij"))

        object.__setattr__(self, "_modal_axes", sh.modal_axes)
        object.__setattr__(self, "_modal_shape", sh.modal_shape)
        object.__setattr__(self, "_modal_padding", sh.modal_padding)
        object.__setattr__(self, "_mask", sh.mask)
        object.__setattr__(self, "_modal_mesh",
                           np.meshgrid(*sh.modal_axes, indexing="ij"))

        _cos_lat = np.sqrt(1 - sin_lat**2)
        object.__setattr__(self, "_cos_lat", _cos_lat)
        object.__setattr__(self, "_sec2_lat", 1 / (1 - sin_lat**2))

        _, l = self.modal_axes
        object.__setattr__(self, "_laplacian_eigenvalues",
                           -l * (l + 1) / (self.radius**2))

        # Pre-compute derivative recurrence weights
        m_wt, l_wt = self.modal_mesh
        mask_wt = self.mask.astype(float)
        a = np.sqrt(mask_wt * (l_wt**2 - m_wt**2) / (4 * l_wt**2 - 1))
        a[:, 0] = 0
        b = np.sqrt(mask_wt * ((l_wt + 1)**2 - m_wt**2) / (4 * (l_wt + 1)**2 - 1))
        b[:, -1] = 0
        object.__setattr__(self, "_derivative_recurrence_weights_val", (a, b))

        # Register static tensors as nn.Module buffers on self._buf so
        # Dynamo natively traces buffer access with correct device
        # propagation — no graph breaks, no @torch.compiler.disable.
        _buf = _GridBuffers()
        _buf._init_cos_lat_torch = torch.from_numpy(_cos_lat.copy()).to(device=_dev, dtype=torch.float32)
        _buf._init_sec2_lat_torch = torch.from_numpy((1 / (1 - sin_lat**2)).copy()).to(device=_dev, dtype=torch.float32)
        _buf._init_laplacian_eigenvalues_torch = torch.from_numpy((-l * (l + 1) / (self.radius**2)).copy()).to(device=_dev, dtype=torch.float32)
        _buf._init_deriv_a_torch = torch.from_numpy(a.copy()).to(device=_dev, dtype=torch.float32)
        _buf._init_deriv_b_torch = torch.from_numpy(b.copy()).to(device=_dev, dtype=torch.float32)
        _buf._init_modal_l_torch = torch.from_numpy(l.copy()).to(device=_dev, dtype=torch.float32)
        lon, sin_lat = sh.nodal_axes
        _buf._init_lon_torch = torch.from_numpy(lon.copy()).to(device=_dev, dtype=torch.float32)
        _buf._init_sin_lat_torch = torch.from_numpy(sin_lat.copy()).to(device=_dev, dtype=torch.float32)
        _buf._finalize()
        self._buf = _buf

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        return self is other

    def to(self, device: torch.device | str) -> None:
        """Move all static tensors to *device* (in-place)."""
        _device = torch.device(device)
        self._buf.to(device=_device)
        self._spherical_harmonics._basis.to(device=_device)

    @classmethod
    def with_wavenumbers(
        cls,
        longitude_wavenumbers: int,
        dealiasing: str = "quadratic",
        latitude_spacing: str = "gauss",
        longitude_offset: float = 0.0,
        spherical_harmonics_impl: SphericalHarmonicsImpl = RealSphericalHarmonics,
        radius: float = 1.0,
    ) -> Grid:
        """Construct Grid by specifying only wavenumbers."""
        order = {"linear": 2, "quadratic": 3, "cubic": 4}[dealiasing]
        longitude_nodes = order * longitude_wavenumbers + 1
        latitude_nodes = math.ceil(longitude_nodes / 2)
        return cls(
            longitude_wavenumbers=longitude_wavenumbers,
            total_wavenumbers=longitude_wavenumbers + 1,
            longitude_nodes=longitude_nodes,
            latitude_nodes=latitude_nodes,
            latitude_spacing=latitude_spacing,
            longitude_offset=longitude_offset,
            spherical_harmonics_impl=spherical_harmonics_impl,
            radius=radius,
        )

    @classmethod
    def construct(
        cls,
        max_wavenumber: int,
        gaussian_nodes: int,
        latitude_spacing: str = "gauss",
        longitude_offset: float = 0.0,
        radius: float = 1.0,
        spherical_harmonics_impl: SphericalHarmonicsImpl = RealSphericalHarmonics,
    ) -> Grid:
        """Construct Grid by specifying max wavenumber & Gaussian nodes."""
        return cls(
            longitude_wavenumbers=max_wavenumber + 1,
            total_wavenumbers=max_wavenumber + 2,
            longitude_nodes=4 * gaussian_nodes,
            latitude_nodes=2 * gaussian_nodes,
            latitude_spacing=latitude_spacing,
            longitude_offset=longitude_offset,
            spherical_harmonics_impl=spherical_harmonics_impl,
            radius=radius,
        )

    # ── Standard grid factory methods ─────────────────────────────────────

    @classmethod
    def T21(cls, **kw) -> Grid: return cls.construct(21, 16, **kw)
    @classmethod
    def T31(cls, **kw) -> Grid: return cls.construct(31, 24, **kw)
    @classmethod
    def T42(cls, **kw) -> Grid: return cls.construct(42, 32, **kw)
    @classmethod
    def T85(cls, **kw) -> Grid: return cls.construct(85, 64, **kw)
    @classmethod
    def T106(cls, **kw) -> Grid: return cls.construct(106, 80, **kw)
    @classmethod
    def T119(cls, **kw) -> Grid: return cls.construct(119, 90, **kw)
    @classmethod
    def T170(cls, **kw) -> Grid: return cls.construct(170, 128, **kw)
    @classmethod
    def T213(cls, **kw) -> Grid: return cls.construct(213, 160, **kw)
    @classmethod
    def T340(cls, **kw) -> Grid: return cls.construct(340, 256, **kw)
    @classmethod
    def T425(cls, **kw) -> Grid: return cls.construct(425, 320, **kw)
    @classmethod
    def TL31(cls, **kw) -> Grid: return cls.construct(31, 16, **kw)
    @classmethod
    def TL47(cls, **kw) -> Grid: return cls.construct(47, 24, **kw)
    @classmethod
    def TL63(cls, **kw) -> Grid: return cls.construct(63, 32, **kw)
    @classmethod
    def TL95(cls, **kw) -> Grid: return cls.construct(95, 48, **kw)
    @classmethod
    def TL127(cls, **kw) -> Grid: return cls.construct(127, 64, **kw)
    @classmethod
    def TL159(cls, **kw) -> Grid: return cls.construct(159, 80, **kw)
    @classmethod
    def TL179(cls, **kw) -> Grid: return cls.construct(179, 90, **kw)
    @classmethod
    def TL255(cls, **kw) -> Grid: return cls.construct(255, 128, **kw)
    @classmethod
    def TL639(cls, **kw) -> Grid: return cls.construct(639, 320, **kw)
    @classmethod
    def TL1279(cls, **kw) -> Grid: return cls.construct(1279, 640, **kw)

    # ── Properties ────────────────────────────────────────────────────────
    # All former @functools.cached_property values are pre-computed in
    # __post_init__ and exposed as plain @property so torch.compile
    # (Dynamo) never sees a cached_property write on a frozen dataclass.

    def with_einsum(self, einsum_fn) -> "Grid":
        """Return a new Grid that uses a custom einsum for SHT.

        Useful for forcing a specific precision in transforms, e.g.::

            grid_fp64 = grid.with_einsum(einsum_highest)
            grid_fast = grid.with_einsum(einsum_adaptive)

        The new Grid inherits the source Grid's device via ``_initial_device``
        so all static tensors and basis matrices are created directly on the
        target device — no post-creation ``.to(device=…)`` needed and no
        Dynamo graph break.
        """
        return dataclasses.replace(
            self,
            spherical_harmonics_impl=functools.partial(
                self.spherical_harmonics_impl, einsum_fn=einsum_fn
            ),
            _initial_device=self._buf._cos_lat_torch.device,
        )

    @property
    def spherical_harmonics(self) -> RealSphericalHarmonics:
        return self._spherical_harmonics

    @property
    def longitudes(self) -> np.ndarray:
        return self.nodal_axes[0]

    @property
    def latitudes(self) -> np.ndarray:
        return np.arcsin(self.nodal_axes[1])

    @property
    def nodal_axes(self) -> tuple[np.ndarray, np.ndarray]:
        return self._nodal_axes

    @property
    def nodal_shape(self) -> tuple[int, int]:
        return self._nodal_shape

    @property
    def nodal_padding(self) -> tuple[int, int]:
        return self._nodal_padding

    @property
    def nodal_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        return self._nodal_mesh

    @property
    def modal_axes(self) -> tuple[np.ndarray, np.ndarray]:
        return self._modal_axes

    @property
    def modal_shape(self) -> tuple[int, int]:
        return self._modal_shape

    @property
    def modal_padding(self) -> tuple[int, int]:
        return self._modal_padding

    @property
    def mask(self) -> np.ndarray:
        return self._mask

    @property
    def modal_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        return self._modal_mesh

    @property
    def cos_lat(self) -> np.ndarray:
        return self._cos_lat

    @property
    def sec2_lat(self) -> np.ndarray:
        return self._sec2_lat

    @property
    def laplacian_eigenvalues(self) -> np.ndarray:
        return self._laplacian_eigenvalues

    # ── Transforms ────────────────────────────────────────────────────────

    def to_nodal(self, x: torch.Tensor) -> torch.Tensor:
        """Modal → nodal transform."""
        return self.spherical_harmonics.inverse_transform(x)

    def to_modal(self, x: torch.Tensor) -> torch.Tensor:
        """Nodal → modal transform."""
        return self.spherical_harmonics.transform(x)

    # ── Spectral operators ────────────────────────────────────────────────

    def laplacian(self, x: torch.Tensor) -> torch.Tensor:
        """∇²(x) in spectral basis."""
        eig = self._buf._laplacian_eigenvalues_torch.to(device=x.device, dtype=x.dtype)
        return x * eig

    def inverse_laplacian(self, x: torch.Tensor) -> torch.Tensor:
        """(∇²)⁻¹(x) in spectral basis."""
        eig = self._buf._laplacian_eigenvalues_torch
        inv_eig = 1.0 / eig.to(device=x.device, dtype=x.dtype)
        # Functional mask: zero out l=0 (mean) and wavenumbers beyond truncation.
        indices = torch.arange(len(inv_eig), device=x.device)
        wavenum_mask = (indices != 0) & (indices < self.total_wavenumbers)
        inv_eig = torch.where(wavenum_mask, inv_eig, torch.zeros_like(inv_eig))
        return x * inv_eig

    def clip_wavenumbers(self, x, n: int = 1):
        """Zero out the highest n total wavenumbers."""
        if n <= 0:
            raise ValueError(f"n must be >= 0; got {n}.")

        def clip_fn(t):
            if not isinstance(t, torch.Tensor):
                return t
            num_zeros = n + self.modal_padding[-1]
            L = self.modal_shape[-1]
            indices = torch.arange(L, device=t.device)
            mask = (indices < L - num_zeros).to(dtype=t.dtype)
            return t * mask

        if isinstance(x, tuple):
            return tuple(clip_fn(xi) for xi in x)
        return clip_fn(x)

    @property
    def _derivative_recurrence_weights(self) -> tuple[np.ndarray, np.ndarray]:
        return self._derivative_recurrence_weights_val

    def _shift(self, x: torch.Tensor, offset: int, axis: int) -> torch.Tensor:
        """Circular shift along axis."""
        n = x.shape[axis]
        dim = axis if axis >= 0 else x.ndim + axis
        idx = (torch.arange(n, device=x.device) - offset) % n
        return x.index_select(dim, idx)

    def d_dlon(self, x: torch.Tensor) -> torch.Tensor:
        """∂x/∂λ in modal basis."""
        return self.spherical_harmonics.longitudinal_derivative(x)

    def cos_lat_d_dlat(self, x: torch.Tensor) -> torch.Tensor:
        """cos(θ) ∂x/∂θ in modal basis."""
        a = self._buf._deriv_a_torch.to(device=x.device, dtype=x.dtype)
        b = self._buf._deriv_b_torch.to(device=x.device, dtype=x.dtype)
        l_t = self._buf._modal_l_torch.to(device=x.device, dtype=x.dtype)
        x_lm1 = self._shift(((l_t + 1) * a) * x, -1, axis=-1)
        x_lp1 = self._shift((-l_t * b) * x, +1, axis=-1)
        return x_lm1 + x_lp1

    def sec_lat_d_dlat_cos2(self, x: torch.Tensor) -> torch.Tensor:
        """sec(θ) ∂/∂θ(cos²θ x) in modal basis."""
        a = self._buf._deriv_a_torch.to(device=x.device, dtype=x.dtype)
        b = self._buf._deriv_b_torch.to(device=x.device, dtype=x.dtype)
        l_t = self._buf._modal_l_torch.to(device=x.device, dtype=x.dtype)
        x_lm1 = self._shift(((l_t - 1) * a) * x, -1, axis=-1)
        x_lp1 = self._shift((-(l_t + 2) * b) * x, +1, axis=-1)
        return x_lm1 + x_lp1

    def cos_lat_grad(
        self, x: torch.Tensor, clip: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """cos(θ) ∇(x)."""
        raw = (
            self.d_dlon(x) / self.radius,
            self.cos_lat_d_dlat(x) / self.radius,
        )
        if clip:
            return self.clip_wavenumbers(raw)
        return raw

    def k_cross(
        self, v: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """k × v where k is the normal unit vector."""
        return (-v[1], v[0])

    def div_cos_lat(
        self,
        v: tuple[torch.Tensor, torch.Tensor],
        clip: bool = True,
    ) -> torch.Tensor:
        """∇ · (v cos θ)."""
        raw = (self.d_dlon(v[0]) + self.sec_lat_d_dlat_cos2(v[1])) / self.radius
        if clip:
            return self.clip_wavenumbers(raw)
        return raw

    def curl_cos_lat(
        self,
        v: tuple[torch.Tensor, torch.Tensor],
        clip: bool = True,
    ) -> torch.Tensor:
        """k · ∇ × (v cos θ)."""
        raw = (self.d_dlon(v[1]) - self.sec_lat_d_dlat_cos2(v[0])) / self.radius
        if clip:
            return self.clip_wavenumbers(raw)
        return raw

    @property
    def quadrature_weights(self) -> torch.Tensor:
        return self.spherical_harmonics.basis.w.broadcast_to(self.nodal_shape)

    def integrate(self, z: torch.Tensor) -> torch.Tensor:
        """Integrate nodal values over the sphere."""
        w = self.spherical_harmonics.basis.w.to(device=z.device, dtype=z.dtype)
        w = w * (self.radius ** 2)
        return torch.einsum("y,...xy->...", w, z)


# Register Grid as a pytree node so Dynamo traces its tensor children with
# correct FakeTensor device propagation.  Without this, getattr(grid, attr)
# returns CPU FakeTensors that cause "Unhandled FakeTensor Device Propagation"
# when mixed with CUDA input FakeTensors.
def __grid_flatten(grid: Grid):
    children = (
        grid._buf,                # _GridBuffers (nn.Module with 8 buffers)
        grid._spherical_harmonics,  # RealSphericalHarmonics — pytree-registered
    )
    context = {
        "longitude_wavenumbers": grid.longitude_wavenumbers,
        "total_wavenumbers": grid.total_wavenumbers,
        "longitude_nodes": grid.longitude_nodes,
        "latitude_nodes": grid.latitude_nodes,
        "latitude_spacing": grid.latitude_spacing,
        "longitude_offset": grid.longitude_offset,
        "radius": grid.radius,
        "spherical_harmonics_impl": grid.spherical_harmonics_impl,
        "_initial_device": grid._initial_device,
    }
    return children, context


def __grid_unflatten(children, context):
    buf, spherical_harmonics = children
    grid = Grid(**context)
    object.__setattr__(grid, "_buf", buf)
    object.__setattr__(grid, "_spherical_harmonics", spherical_harmonics)
    return grid


_reg_pytree(Grid, __grid_flatten, __grid_unflatten)


# ═══════════════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════════════

def add_constant(x: torch.Tensor, c: float) -> torch.Tensor:
    """Add constant c to spectral array x (modifies [0, 0] coefficient)."""
    x = x.clone()
    x[..., 0, 0] = x[..., 0, 0] + _CONSTANT_NORMALIZATION_FACTOR * c
    return x


def get_cos_lat_vector(
    vorticity: torch.Tensor,
    divergence: torch.Tensor,
    grid: Grid,
    clip: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute v cos(θ) from vorticity and divergence."""
    stream_function = grid.inverse_laplacian(vorticity)
    velocity_potential = grid.inverse_laplacian(divergence)
    grad_vp = grid.cos_lat_grad(velocity_potential, clip=clip)
    k_cross_grad_sf = grid.k_cross(
        grid.cos_lat_grad(stream_function, clip=clip)
    )
    return (grad_vp[0] + k_cross_grad_sf[0], grad_vp[1] + k_cross_grad_sf[1])


def uv_nodal_to_vor_div_modal(
    grid: Grid,
    u_nodal: torch.Tensor,
    v_nodal: torch.Tensor,
    clip: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert nodal u,v velocities to modal vorticity, divergence."""
    cos_lat = grid._buf._cos_lat_torch.to(device=u_nodal.device, dtype=u_nodal.dtype)
    u_over_cos_lat = grid.to_modal(u_nodal / cos_lat)
    v_over_cos_lat = grid.to_modal(v_nodal / cos_lat)
    vorticity = grid.curl_cos_lat((u_over_cos_lat, v_over_cos_lat), clip=clip)
    divergence = grid.div_cos_lat((u_over_cos_lat, v_over_cos_lat), clip=clip)
    return vorticity, divergence


def vor_div_to_uv_nodal(
    grid: Grid,
    vorticity: torch.Tensor,
    divergence: torch.Tensor,
    clip: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert modal vorticity, divergence to nodal u,v velocities."""
    u_cos_lat, v_cos_lat = get_cos_lat_vector(
        vorticity, divergence, grid, clip=clip
    )
    cos_lat = grid._buf._cos_lat_torch.to(device=u_cos_lat.device, dtype=u_cos_lat.dtype)
    u_nodal = grid.to_nodal(u_cos_lat) / cos_lat
    v_nodal = grid.to_nodal(v_cos_lat) / cos_lat
    return u_nodal, v_nodal
