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

from pytorch_src.core import associated_legendre
from pytorch_src.core import fourier
from pytorch_src.precision.zone_cast import einsum_highest


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

@dataclasses.dataclass
class _SphericalHarmonicBasis:
    """Precomputed basis matrices for spherical harmonic transforms."""
    f: np.ndarray  # Fourier matrix (float64)
    p: np.ndarray  # Legendre coefficients (float64)
    w: np.ndarray  # Quadrature weights (float64)


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

    @functools.cached_property
    def _einsum(self):
        """Resolve the einsum function (Algorithm 8: adaptive precision)."""
        if self.einsum_fn is not None:
            return self.einsum_fn
        return get_sht_einsum(self.total_wavenumbers)

    @functools.cached_property
    def nodal_axes(self) -> tuple[np.ndarray, np.ndarray]:
        longitude, _ = fourier.quadrature_nodes(self.longitude_nodes)
        sin_latitude, _ = get_latitude_nodes(
            self.latitude_nodes, self.latitude_spacing
        )
        return longitude, sin_latitude

    @functools.cached_property
    def nodal_shape(self) -> tuple[int, int]:
        return (self.longitude_nodes, self.latitude_nodes)

    @functools.cached_property
    def nodal_padding(self) -> tuple[int, int]:
        return (0, 0)

    @functools.cached_property
    def modal_axes(self) -> tuple[np.ndarray, np.ndarray]:
        m_pos = np.arange(1, self.longitude_wavenumbers)
        m_pos_neg = np.stack([m_pos, -m_pos], axis=1).ravel()
        lon_wavenumbers = np.concatenate([[0], m_pos_neg])
        tot_wavenumbers = np.arange(self.total_wavenumbers)
        return lon_wavenumbers, tot_wavenumbers

    @functools.cached_property
    def modal_shape(self) -> tuple[int, int]:
        return (2 * self.longitude_wavenumbers - 1, self.total_wavenumbers)

    @functools.cached_property
    def modal_padding(self) -> tuple[int, int]:
        return (0, 0)

    @functools.cached_property
    def modal_dtype(self) -> np.dtype:
        return np.dtype(np.float32)

    @functools.cached_property
    def mask(self) -> np.ndarray:
        m, l = np.meshgrid(*self.modal_axes, indexing="ij")
        return abs(m) <= l

    @functools.cached_property
    def basis(self) -> _SphericalHarmonicBasis:
        f = fourier.real_basis(
            wavenumbers=self.longitude_wavenumbers,
            nodes=self.longitude_nodes,
        )
        _, wf = fourier.quadrature_nodes(self.longitude_nodes)
        x, wp = get_latitude_nodes(self.latitude_nodes, self.latitude_spacing)
        w = wf * wp
        p = associated_legendre.evaluate(
            n_m=self.longitude_wavenumbers,
            n_l=self.total_wavenumbers,
            x=x,
        )
        # Duplicate rows for cos/sin pairing
        p = np.repeat(p, 2, axis=0)
        p = p[1:]  # Remove extra m=0 duplicate
        return _SphericalHarmonicBasis(f=f, p=p, w=w)

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Modal → nodal transform."""
        _einsum = self._einsum
        p = torch.tensor(self.basis.p, dtype=torch.float64, device=x.device)
        f = torch.tensor(self.basis.f, dtype=torch.float64, device=x.device)
        # Legendre: mjl,...ml -> ...mj
        px = _einsum("mjl,...ml->...mj", p, x)
        # Fourier: im,...mj -> ...ij
        fpx = _einsum("im,...mj->...ij", f, px)
        return fpx

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Nodal → modal transform."""
        _einsum = self._einsum
        w = torch.tensor(self.basis.w, dtype=x.dtype, device=x.device)
        p = torch.tensor(self.basis.p, dtype=torch.float64, device=x.device)
        f = torch.tensor(self.basis.f, dtype=torch.float64, device=x.device)
        wx = w * x
        # Fourier: im,...ij -> ...mj
        fwx = _einsum("im,...ij->...mj", f, wx)
        # Legendre: mjl,...mj -> ...ml
        pfwx = _einsum("mjl,...mj->...ml", p, fwx)
        return pfwx

    def longitudinal_derivative(self, x: torch.Tensor) -> torch.Tensor:
        """∂x/∂λ in modal basis."""
        return fourier.real_basis_derivative(x, axis=-2)


# ═══════════════════════════════════════════════════════════════════════════
# Grid — the main user-facing class
# ═══════════════════════════════════════════════════════════════════════════

SphericalHarmonicsImpl = Callable[..., RealSphericalHarmonics]

_CONSTANT_NORMALIZATION_FACTOR = 3.5449077


@dataclasses.dataclass(frozen=True)
class Grid:
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

    def __post_init__(self):
        if self.radius is None:
            object.__setattr__(self, "radius", 1.0)
        if self.latitude_spacing not in LATITUDE_SPACINGS:
            raise ValueError(
                f'Unsupported latitude_spacing "{self.latitude_spacing}". '
                f"Supported: {list(LATITUDE_SPACINGS)}."
            )

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

    @functools.cached_property
    def spherical_harmonics(self) -> RealSphericalHarmonics:
        return self.spherical_harmonics_impl(
            longitude_wavenumbers=self.longitude_wavenumbers,
            total_wavenumbers=self.total_wavenumbers,
            longitude_nodes=self.longitude_nodes,
            latitude_nodes=self.latitude_nodes,
            latitude_spacing=self.latitude_spacing,
            einsum_fn=get_sht_einsum(self.total_wavenumbers),
        )

    def with_einsum(self, einsum_fn) -> "Grid":
        """Return a new Grid that uses a custom einsum for SHT.

        Useful for forcing a specific precision in transforms, e.g.::

            grid_fp64 = grid.with_einsum(einsum_highest)
            grid_fast = grid.with_einsum(einsum_adaptive)
        """
        return dataclasses.replace(
            self,
            spherical_harmonics_impl=functools.partial(
                self.spherical_harmonics_impl, einsum_fn=einsum_fn
            ),
        )

    @property
    def longitudes(self) -> np.ndarray:
        return self.nodal_axes[0]

    @property
    def latitudes(self) -> np.ndarray:
        return np.arcsin(self.nodal_axes[1])

    @functools.cached_property
    def nodal_axes(self) -> tuple[np.ndarray, np.ndarray]:
        lon, sin_lat = self.spherical_harmonics.nodal_axes
        return lon + self.longitude_offset, sin_lat

    @functools.cached_property
    def nodal_shape(self) -> tuple[int, int]:
        return self.spherical_harmonics.nodal_shape

    @functools.cached_property
    def nodal_padding(self) -> tuple[int, int]:
        return self.spherical_harmonics.nodal_padding

    @functools.cached_property
    def nodal_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(*self.nodal_axes, indexing="ij")

    @functools.cached_property
    def modal_axes(self) -> tuple[np.ndarray, np.ndarray]:
        return self.spherical_harmonics.modal_axes

    @functools.cached_property
    def modal_shape(self) -> tuple[int, int]:
        return self.spherical_harmonics.modal_shape

    @functools.cached_property
    def modal_padding(self) -> tuple[int, int]:
        return self.spherical_harmonics.modal_padding

    @functools.cached_property
    def mask(self) -> np.ndarray:
        return self.spherical_harmonics.mask

    @functools.cached_property
    def modal_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(*self.spherical_harmonics.modal_axes, indexing="ij")

    @functools.cached_property
    def cos_lat(self) -> np.ndarray:
        _, sin_lat = self.nodal_axes
        return np.sqrt(1 - sin_lat**2)

    @functools.cached_property
    def sec2_lat(self) -> np.ndarray:
        _, sin_lat = self.nodal_axes
        return 1 / (1 - sin_lat**2)

    @functools.cached_property
    def laplacian_eigenvalues(self) -> np.ndarray:
        _, l = self.modal_axes
        return -l * (l + 1) / (self.radius**2)

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
        eig = torch.tensor(
            self.laplacian_eigenvalues, dtype=x.dtype, device=x.device
        )
        return x * eig

    def inverse_laplacian(self, x: torch.Tensor) -> torch.Tensor:
        """(∇²)⁻¹(x) in spectral basis."""
        eig = self.laplacian_eigenvalues.copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_eig = 1 / eig
        inv_eig[0] = 0
        inv_eig[self.total_wavenumbers:] = 0
        inv_eig_t = torch.tensor(inv_eig, dtype=x.dtype, device=x.device)
        return x * inv_eig_t

    def clip_wavenumbers(self, x, n: int = 1):
        """Zero out the highest n total wavenumbers."""
        if n <= 0:
            raise ValueError(f"n must be >= 0; got {n}.")

        def clip_fn(t):
            if not isinstance(t, torch.Tensor):
                return t
            num_zeros = n + self.modal_padding[-1]
            mask = torch.ones(
                self.modal_shape[-1], dtype=t.dtype, device=t.device
            )
            mask[-num_zeros:] = 0
            return t * mask

        if isinstance(x, tuple):
            return tuple(clip_fn(xi) for xi in x)
        return clip_fn(x)

    @functools.cached_property
    def _derivative_recurrence_weights(self) -> tuple[np.ndarray, np.ndarray]:
        m, l = self.modal_mesh
        mask = self.mask.astype(float)
        a = np.sqrt(mask * (l**2 - m**2) / (4 * l**2 - 1))
        a[:, 0] = 0
        b = np.sqrt(mask * ((l + 1) ** 2 - m**2) / (4 * (l + 1) ** 2 - 1))
        b[:, -1] = 0
        return a, b

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
        _, l = self.modal_mesh
        a_np, b_np = self._derivative_recurrence_weights
        a = torch.tensor(a_np, dtype=x.dtype, device=x.device)
        b = torch.tensor(b_np, dtype=x.dtype, device=x.device)
        l_t = torch.tensor(l, dtype=x.dtype, device=x.device)
        x_lm1 = self._shift(((l_t + 1) * a) * x, -1, axis=-1)
        x_lp1 = self._shift((-l_t * b) * x, +1, axis=-1)
        return x_lm1 + x_lp1

    def sec_lat_d_dlat_cos2(self, x: torch.Tensor) -> torch.Tensor:
        """sec(θ) ∂/∂θ(cos²θ x) in modal basis."""
        _, l = self.modal_mesh
        a_np, b_np = self._derivative_recurrence_weights
        a = torch.tensor(a_np, dtype=x.dtype, device=x.device)
        b = torch.tensor(b_np, dtype=x.dtype, device=x.device)
        l_t = torch.tensor(l, dtype=x.dtype, device=x.device)
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
    def quadrature_weights(self) -> np.ndarray:
        return np.broadcast_to(
            self.spherical_harmonics.basis.w, self.nodal_shape
        )

    def integrate(self, z: torch.Tensor) -> torch.Tensor:
        """Integrate nodal values over the sphere."""
        w = torch.tensor(
            self.spherical_harmonics.basis.w * self.radius**2,
            dtype=z.dtype, device=z.device,
        )
        return torch.einsum("y,...xy->...", w, z)


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
    cos_lat = torch.tensor(grid.cos_lat, dtype=u_nodal.dtype, device=u_nodal.device)
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
    cos_lat = torch.tensor(grid.cos_lat, dtype=u_cos_lat.dtype, device=u_cos_lat.device)
    u_nodal = grid.to_nodal(u_cos_lat) / cos_lat
    v_nodal = grid.to_nodal(v_cos_lat) / cos_lat
    return u_nodal, v_nodal
