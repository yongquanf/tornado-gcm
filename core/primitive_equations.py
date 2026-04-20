# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""Primitive equations data structures, diagnostics, and RHS — PyTorch.

This module provides:
  - State dataclass for spectral primitive equations
  - DiagnosticState computation
  - Full explicit/implicit/implicit_inverse terms for sigma coords
  - PrimitiveEquationsSigma (ImplicitExplicitODE)
  - Helper functions: geopotential, temperature implicit weights, etc.

Zone: Z1 (dynamics compute) for equation arithmetic.
Zone: Z0 for einsum via einsum_highest.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Mapping, Optional

import numpy as np
import torch

from tornado_gcm.core import coordinate_systems
from tornado_gcm.core import sigma_coordinates as sigma_mod
from tornado_gcm.core import spherical_harmonic
from tornado_gcm.core import time_integration
from tornado_gcm.precision.zone_cast import einsum_highest


@dataclasses.dataclass
class State:
    """State of the primitive equations (modal / spectral representation).

    Attributes:
        vorticity: modal vorticity field (layers, m, l).
        divergence: modal divergence field (layers, m, l).
        temperature_variation: modal T' field (layers, m, l).
        log_surface_pressure: modal log(Ps) (1, m, l).
        tracers: dict of tracer fields (layers, m, l).
        sim_time: simulation time (scalar or None).
    """

    vorticity: torch.Tensor
    divergence: torch.Tensor
    temperature_variation: torch.Tensor
    log_surface_pressure: torch.Tensor
    tracers: dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)
    sim_time: float | None = None

    def __add__(self, other: State) -> State:
        tracers = {k: self.tracers[k] + other.tracers[k] for k in self.tracers}
        return State(
            vorticity=self.vorticity + other.vorticity,
            divergence=self.divergence + other.divergence,
            temperature_variation=self.temperature_variation + other.temperature_variation,
            log_surface_pressure=self.log_surface_pressure + other.log_surface_pressure,
            tracers=tracers,
            sim_time=self.sim_time,
        )

    def __mul__(self, scalar: float) -> State:
        tracers = {k: v * scalar for k, v in self.tracers.items()}
        return State(
            vorticity=self.vorticity * scalar,
            divergence=self.divergence * scalar,
            temperature_variation=self.temperature_variation * scalar,
            log_surface_pressure=self.log_surface_pressure * scalar,
            tracers=tracers,
            sim_time=self.sim_time,
        )

    def __rmul__(self, scalar: float) -> State:
        return self.__mul__(scalar)

    def tree_map(self, fn):
        tracers = {k: fn(v) for k, v in self.tracers.items()}
        return State(
            vorticity=fn(self.vorticity),
            divergence=fn(self.divergence),
            temperature_variation=fn(self.temperature_variation),
            log_surface_pressure=fn(self.log_surface_pressure),
            tracers=tracers,
            sim_time=self.sim_time,
        )


class StateShapeError(Exception):
    pass


def validate_state_shape(
    state: State, coords: coordinate_systems.CoordinateSystem
):
    """Validate that state shapes are consistent with coords."""
    if state.vorticity.shape != coords.modal_shape:
        raise StateShapeError(
            f"Expected vorticity shape {coords.modal_shape}; "
            f"got {state.vorticity.shape}."
        )
    if state.divergence.shape != coords.modal_shape:
        raise StateShapeError(
            f"Expected divergence shape {coords.modal_shape}; "
            f"got {state.divergence.shape}."
        )
    if state.temperature_variation.shape != coords.modal_shape:
        raise StateShapeError(
            f"Expected temperature_variation shape {coords.modal_shape}; "
            f"got {state.temperature_variation.shape}."
        )
    if state.log_surface_pressure.shape != coords.surface_modal_shape:
        raise StateShapeError(
            f"Expected log_surface_pressure shape {coords.surface_modal_shape}; "
            f"got {state.log_surface_pressure.shape}."
        )


def vertical_matvec(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Matrix-vector product along the vertical axis: a @ x per (m,l)."""
    return einsum_highest("gh,...hml->...gml", a, x)


# ═══════════════════════════════════════════════════════════════════════════
# Diagnostic state
# ═══════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class DiagnosticState:
    """Nodal diagnostic values for computing explicit tendencies.

    All fields are in nodal (physical) space.
    """

    vorticity: torch.Tensor
    divergence: torch.Tensor
    temperature_variation: torch.Tensor
    cos_lat_u: tuple[torch.Tensor, torch.Tensor]
    sigma_dot_explicit: torch.Tensor
    sigma_dot_full: torch.Tensor
    cos_lat_grad_log_sp: tuple[torch.Tensor, torch.Tensor]
    u_dot_grad_log_sp: torch.Tensor
    tracers: dict[str, torch.Tensor]


def compute_diagnostic_state(
    state: State,
    coords: coordinate_systems.CoordinateSystem,
) -> DiagnosticState:
    """Compute nodal diagnostic state from modal State.

    This transforms modal fields to nodal space and evaluates key diagnostic
    quantities needed for the explicit tendency computation.
    """
    # Align with dinosaur/JAX high-precision SH transform behavior in diagnostic
    # construction to reduce to_nodal / cos_lat_grad drift on low-resolution grids.
    grid = coords.horizontal.with_einsum(einsum_highest)
    out_dtype = state.vorticity.dtype
    modal_work_dtype = torch.float64

    vorticity_modal = state.vorticity.to(modal_work_dtype)
    divergence_modal = state.divergence.to(modal_work_dtype)
    temperature_modal = state.temperature_variation.to(modal_work_dtype)
    log_sp_modal = state.log_surface_pressure.to(modal_work_dtype)
    tracers_modal = {k: v.to(modal_work_dtype) for k, v in state.tracers.items()}

    nodal_vorticity_work = grid.to_nodal(vorticity_modal)
    nodal_divergence_work = grid.to_nodal(divergence_modal)
    nodal_temperature_variation_work = grid.to_nodal(temperature_modal)

    # Tracers
    nodal_tracers_work = {k: grid.to_nodal(v) for k, v in tracers_modal.items()}

    # cos(lat) * velocity from vorticity + divergence
    cos_lat_u_modal = spherical_harmonic.get_cos_lat_vector(
        vorticity_modal, divergence_modal, grid, clip=False
    )
    nodal_cos_lat_u_work = (
        grid.to_nodal(cos_lat_u_modal[0]),
        grid.to_nodal(cos_lat_u_modal[1]),
    )

    # cos(lat) * grad(log(Ps))
    cos_lat_grad_log_sp_modal = grid.cos_lat_grad(
        log_sp_modal, clip=False
    )
    nodal_cos_lat_grad_log_sp_work = (
        grid.to_nodal(cos_lat_grad_log_sp_modal[0]),
        grid.to_nodal(cos_lat_grad_log_sp_modal[1]),
    )

    # u · grad(log(Ps))
    # Keep the same operation order as dinosaur/JAX:
    #   (u*grad_u*sec2) + (v*grad_v*sec2)
    # This is more numerically stable under cancellation than
    #   (u*grad_u + v*grad_v) * sec2
    # and reduces PT/JAX drift in sigma_dot diagnostics.
    work_dtype = torch.float64
    sec2_lat = torch.tensor(
        grid.sec2_lat, dtype=work_dtype, device=nodal_cos_lat_u_work[0].device,
    )
    u_comp = nodal_cos_lat_u_work[0].to(work_dtype)
    v_comp = nodal_cos_lat_u_work[1].to(work_dtype)
    grad_u = nodal_cos_lat_grad_log_sp_work[0].to(work_dtype)
    grad_v = nodal_cos_lat_grad_log_sp_work[1].to(work_dtype)
    u_dot_grad_log_sp_work = (
        u_comp * grad_u * sec2_lat
        + v_comp * grad_v * sec2_lat
    )

    # Placeholder sigma_dot — requires vertical coordinate
    # Full computation needs cumulative_sigma_integral which depends on
    # the coordinate system type.
    if hasattr(coords, 'vertical') and hasattr(coords.vertical, 'layer_thickness'):
        vertical = coords.vertical
        f_explicit = sigma_mod.cumulative_sigma_integral(
            u_dot_grad_log_sp_work, vertical, axis=-3, downward=True,
        )
        f_full = sigma_mod.cumulative_sigma_integral(
            nodal_divergence_work + u_dot_grad_log_sp_work,
            vertical,
            axis=-3,
            downward=True,
        )
        # Match dinosaur ``compute_diagnostic_state_sigma``:
        #   sum_σ = cumsum(layer_thickness)[:, None, None]  (length L)
        #   σ̇ = slice_0_L-1( sum_σ * f[..., -1, :, :] - f )
        # (same algebra as former ``cum_delta_sigma[:-1] * total - f[:-1]``, clearer vs lax).
        th = np.asarray(vertical.layer_thickness, dtype=np.float64)
        sum_sigma_1d = torch.tensor(
            np.cumsum(th), dtype=f_explicit.dtype, device=f_explicit.device,
        )
        axis_v = -3

        def _broadcast_sigma_prefactor(t1d: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
            pos = axis_v if axis_v >= 0 else ref.ndim + axis_v
            shp = [1] * ref.ndim
            shp[pos] = int(t1d.shape[0])
            return t1d.reshape(shp)

        sum_sigma_b = _broadcast_sigma_prefactor(sum_sigma_1d, f_explicit)
        f_last_exp = f_explicit.narrow(axis_v, f_explicit.shape[axis_v] - 1, 1)
        f_last_full = f_full.narrow(axis_v, f_full.shape[axis_v] - 1, 1)
        n_sig = int(f_explicit.shape[axis_v]) - 1
        sigma_dot_explicit_work = (
            sum_sigma_b * f_last_exp - f_explicit
        ).narrow(axis_v, 0, n_sig)
        sigma_dot_full_work = (sum_sigma_b * f_last_full - f_full).narrow(axis_v, 0, n_sig)
    else:
        sigma_dot_explicit_work = torch.zeros_like(nodal_divergence_work, dtype=work_dtype)
        sigma_dot_full_work = torch.zeros_like(nodal_divergence_work, dtype=work_dtype)

    nodal_vorticity = nodal_vorticity_work.to(out_dtype)
    nodal_divergence = nodal_divergence_work.to(out_dtype)
    nodal_temperature_variation = nodal_temperature_variation_work.to(out_dtype)
    nodal_cos_lat_u = (
        nodal_cos_lat_u_work[0].to(out_dtype),
        nodal_cos_lat_u_work[1].to(out_dtype),
    )
    nodal_cos_lat_grad_log_sp = (
        nodal_cos_lat_grad_log_sp_work[0].to(out_dtype),
        nodal_cos_lat_grad_log_sp_work[1].to(out_dtype),
    )
    nodal_tracers = {k: v.to(out_dtype) for k, v in nodal_tracers_work.items()}
    u_dot_grad_log_sp = u_dot_grad_log_sp_work.to(out_dtype)
    sigma_dot_explicit = sigma_dot_explicit_work.to(out_dtype)
    sigma_dot_full = sigma_dot_full_work.to(out_dtype)

    return DiagnosticState(
        vorticity=nodal_vorticity,
        divergence=nodal_divergence,
        temperature_variation=nodal_temperature_variation,
        cos_lat_u=nodal_cos_lat_u,
        sigma_dot_explicit=sigma_dot_explicit,
        sigma_dot_full=sigma_dot_full,
        cos_lat_grad_log_sp=nodal_cos_lat_grad_log_sp,
        u_dot_grad_log_sp=u_dot_grad_log_sp,
        tracers=nodal_tracers,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Geopotential / implicit weights helpers (Z0: FP64 precomputation)
# ═══════════════════════════════════════════════════════════════════════════

def get_sigma_ratios(coordinates: sigma_mod.SigmaCoordinates) -> np.ndarray:
    """α_k = log(σ_{k+1/2}) − log(σ_{k-1/2}) / 2, with surface correction."""
    alpha = np.diff(np.log(coordinates.centers), append=0.0).astype(np.float64) / 2.0
    alpha[-1] = -np.log(coordinates.centers[-1])
    return alpha


def get_geopotential_weights_sigma(
    coordinates: sigma_mod.SigmaCoordinates,
    ideal_gas_constant: float,
) -> np.ndarray:
    """Build geopotential weight matrix G (upper triangular).

    G[j,j] = R * α[j]
    G[j,k] = R * (α[k] + α[k-1])  for k > j
    """
    R = ideal_gas_constant
    alpha = get_sigma_ratios(coordinates)
    n = coordinates.layers
    G = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        G[j, j] = R * alpha[j]
        for k in range(j + 1, n):
            G[j, k] = R * (alpha[k] + alpha[k - 1])
    return G


def get_geopotential_diff_sigma(
    temperature: torch.Tensor,
    coordinates: sigma_mod.SigmaCoordinates,
    ideal_gas_constant: float,
) -> torch.Tensor:
    """Φ − Φ_s = G @ T (vertical matrix-vector product)."""
    G = get_geopotential_weights_sigma(coordinates, ideal_gas_constant)
    G_t = torch.tensor(G, dtype=torch.float64, device=temperature.device)
    return einsum_highest("gh,...hml->...gml", G_t, temperature)


def get_geopotential_on_sigma(
    temperature: torch.Tensor,
    nodal_orography: torch.Tensor,
    coordinates: sigma_mod.SigmaCoordinates,
    gravity_acceleration: float,
    ideal_gas_constant: float,
    specific_humidity: Optional[torch.Tensor] = None,
    water_vapor_gas_constant: Optional[float] = None,
) -> torch.Tensor:
    """Full geopotential on sigma levels (nodal space)."""
    surface_geopotential = nodal_orography * gravity_acceleration
    virtual_temp = temperature
    if specific_humidity is not None and water_vapor_gas_constant is not None:
        eps = water_vapor_gas_constant / ideal_gas_constant - 1.0
        virtual_temp = temperature * (1.0 + eps * specific_humidity)
    geo_diff = get_geopotential_diff_sigma(
        virtual_temp, coordinates, ideal_gas_constant,
    )
    return surface_geopotential + geo_diff


def get_temperature_implicit_weights_sigma(
    coordinates: sigma_mod.SigmaCoordinates,
    reference_temperature: np.ndarray,
    kappa: float,
) -> np.ndarray:
    """Build temperature implicit weight matrix H.

    H is needed for the implicit term: dT/dt_implicit = -H @ D
    Based on Durran §8.6.5 discretization.
    """
    T_ref = np.asarray(reference_temperature, dtype=np.float64)
    if T_ref.ndim != 1 or T_ref.shape[-1] != coordinates.layers:
        raise ValueError(
            "reference_temperature must be a vector of length "
            f"coordinates.layers; got shape {T_ref.shape} and "
            f"{coordinates.layers} layers."
        )

    # Match dinosaur.get_temperature_implicit_weights_sigma exactly.
    p = np.tril(np.ones((coordinates.layers, coordinates.layers), dtype=np.float64))

    alpha = get_sigma_ratios(coordinates)[..., np.newaxis]
    p_alpha = p * alpha
    p_alpha_shifted = np.roll(p_alpha, 1, axis=0)
    p_alpha_shifted[0] = 0
    h0 = (
        kappa
        * T_ref[..., np.newaxis]
        * (p_alpha + p_alpha_shifted)
        / coordinates.layer_thickness[..., np.newaxis]
    )

    temp_diff = np.diff(T_ref)
    thickness_sum = coordinates.layer_thickness[:-1] + coordinates.layer_thickness[1:]
    k0 = np.concatenate((temp_diff / thickness_sum, [0.0]), axis=0)[..., np.newaxis]

    thickness_cumulative = np.cumsum(coordinates.layer_thickness)[..., np.newaxis]
    k1 = p - thickness_cumulative
    k = k0 * k1

    k_shifted = np.roll(k, 1, axis=0)
    k_shifted[0] = 0

    return (h0 - k - k_shifted) * coordinates.layer_thickness


def _get_implicit_term_matrix_sigma(
    eta: float,
    coordinates: sigma_mod.SigmaCoordinates,
    reference_temperature: np.ndarray,
    kappa: float,
    ideal_gas_constant: float,
) -> np.ndarray:
    """Build the (2L+1) × (2L+1) block matrix M per wavenumber l.

    M = [[  I,   eta*lambda*G,  eta*R*T_ref  ],
         [ eta*H,     I,          0           ],
         [ eta*Δσ,    0,          1           ]]

    where lambda = -l(l+1)/a² (incorporated externally via eigenvalues).
    Here we build the matrix for scalar multiplication with eig later.
    """
    n = coordinates.layers
    R = ideal_gas_constant
    G = get_geopotential_weights_sigma(coordinates, R)
    H = get_temperature_implicit_weights_sigma(coordinates, reference_temperature, kappa)
    dsigma = coordinates.layer_thickness.astype(np.float64)
    T_ref = np.asarray(reference_temperature, dtype=np.float64)

    dim = 2 * n + 1
    M = np.eye(dim, dtype=np.float64)
    # top-left: I (divergence block)
    # top-mid: eta * G (geopotential)
    M[:n, n:2*n] = eta * G
    # top-right: eta * R * T_ref
    M[:n, 2*n] = eta * R * T_ref
    # mid-left: eta * H (temperature implicit)
    M[n:2*n, :n] = eta * H
    # bottom-left: eta * Δσ
    M[2*n, :n] = eta * dsigma
    return M


def div_sec_lat(
    m_component: torch.Tensor,
    n_component: torch.Tensor,
    grid: spherical_harmonic.Grid,
) -> torch.Tensor:
    """∇·(M,N)/cos²θ in spectral space."""
    sec2 = torch.tensor(
        grid.sec2_lat, dtype=m_component.dtype, device=m_component.device,
    )
    m_sec2 = grid.to_modal(grid.to_nodal(m_component) * sec2)
    n_sec2 = grid.to_modal(grid.to_nodal(n_component) * sec2)
    return grid.div_cos_lat((m_sec2, n_sec2))


def truncated_modal_orography(
    orography: torch.Tensor,
    coords: coordinate_systems.CoordinateSystem,
    wavenumbers_to_clip: int = 1,
) -> torch.Tensor:
    """Orography truncated to spectral resolution with highest wavenumber clipped."""
    grid = coords.horizontal
    modal = grid.to_modal(orography)
    return grid.clip_wavenumbers(modal, n=wavenumbers_to_clip)


def filtered_modal_orography(
    orography: torch.Tensor,
    coords: coordinate_systems.CoordinateSystem,
    filter_fns: tuple[Callable, ...] = (),
) -> torch.Tensor:
    """Orography spectrally interpolated and filtered."""
    grid = coords.horizontal
    modal = grid.to_modal(orography)
    for fn in filter_fns:
        modal = fn(modal)
    return modal


# ═══════════════════════════════════════════════════════════════════════════
# PrimitiveEquationsSigma (IMEX ODE)
# ═══════════════════════════════════════════════════════════════════════════

class PrimitiveEquationsSigma:
    """Full primitive equations in sigma coordinates implementing ImplicitExplicitODE.

    Provides explicit_terms, implicit_terms, implicit_inverse for use with
    IMEX time integrators (e.g. SIL3).

    Attributes:
        reference_temperature: reference temperature profile (layers,). FP64.
        orography: modal orography (m, l). FP32.
        coords: CoordinateSystem (horizontal + vertical).
        physics_specs: SimUnitsProtocol.
        include_vertical_advection: whether to include vertical advection.
        humidity_key: key for specific humidity tracer (None = dry).
        cloud_keys: optional tracer keys for cloud condensates.
    """

    def __init__(
        self,
        reference_temperature: np.ndarray,
        orography: torch.Tensor,
        coords: coordinate_systems.CoordinateSystem,
        physics_specs,
        include_vertical_advection: bool = True,
        humidity_key: Optional[str] = None,
        cloud_keys: Optional[tuple[str, ...]] = None,
    ):
        self.reference_temperature = np.asarray(reference_temperature, dtype=np.float64)
        self.orography = orography
        self.coords = coords
        self.physics_specs = physics_specs
        self.include_vertical_advection = include_vertical_advection
        self.humidity_key = humidity_key
        self.cloud_keys = cloud_keys

        if self.cloud_keys is not None and self.humidity_key is None:
            raise ValueError("cloud_keys requires humidity_key to be set.")

        # Precompute coriolis parameter in nodal space
        _, sin_lat = coords.horizontal.nodal_axes
        self._coriolis = 2.0 * physics_specs.angular_velocity * sin_lat

        # Precompute implicit inverse matrices for each wavenumber
        self._precompute_implicit_inverse()

    @property
    def grid(self) -> spherical_harmonic.Grid:
        return self.coords.horizontal

    @property
    def vertical(self) -> sigma_mod.SigmaCoordinates:
        return self.coords.vertical

    @property
    def T_ref(self) -> np.ndarray:
        return self.reference_temperature

    @property
    def R(self) -> float:
        return self.physics_specs.R

    @property
    def g(self) -> float:
        return self.physics_specs.g

    @property
    def kappa(self) -> float:
        return self.physics_specs.kappa

    def _get_tracer(self, state_or_aux: Any, key: str) -> torch.Tensor:
        tracers = getattr(state_or_aux, "tracers", None)
        if tracers is not None and key in tracers:
            return tracers[key]
        # Synthetic / regridded ICs often carry only ``specific_humidity`` while
        # ``cloud_keys`` still lists condensates for JAX-aligned moist dynamics.
        if self.cloud_keys is not None and key in self.cloud_keys:
            tmpl = getattr(state_or_aux, "temperature_variation", None)
            if tmpl is not None:
                return torch.zeros_like(tmpl)
        available = tuple(tracers.keys()) if tracers is not None else tuple()
        raise ValueError(f"`{key}` is not found in tracers: {available}.")

    def _virtual_temperature_adjustment(self, aux_state: DiagnosticState) -> torch.Tensor:
        """Compute moist virtual-temperature factor, matching JAX path.

        Returns 1 for dry runs; otherwise returns:
          1 + (R_vapor / R_dry - 1) * q - q_cloud
        """
        if self.humidity_key is None:
            return torch.ones_like(aux_state.temperature_variation)

        q = self._get_tracer(aux_state, self.humidity_key)
        gas_const_ratio = self.physics_specs.R_vapor / self.physics_specs.R
        adjustment = 1.0 + (gas_const_ratio - 1.0) * q

        if self.cloud_keys is not None:
            for key in self.cloud_keys:
                # Cloud condensates reduce virtual temperature in JAX reference.
                adjustment = adjustment - self._get_tracer(aux_state, key)

        return adjustment

    def _precompute_implicit_inverse(self):
        """Precompute inverse matrices for implicit_inverse (Z0: FP64)."""
        n = self.vertical.layers
        dim = 2 * n + 1
        L = self.grid.total_wavenumbers
        lam = self.grid.laplacian_eigenvalues  # shape (L,)

        # Store inverse matrices for each l
        self._inv_matrices = []
        for l_idx in range(L):
            # Build M = I - eta * G_l (eta=1 as placeholder, actual eta passed at runtime)
            # We precompute the base matrices; at runtime we build M(eta) and invert
            pass
        # Instead, store the component matrices
        self._G_matrix = get_geopotential_weights_sigma(self.vertical, self.R)
        self._H_matrix = get_temperature_implicit_weights_sigma(
            self.vertical, self.T_ref, self.kappa,
        )
        self._dsigma = self.vertical.layer_thickness.astype(np.float64)

    def kinetic_energy_tendency(
        self,
        aux_state: DiagnosticState,
    ) -> torch.Tensor:
        """−∇²(½|u|²) in modal space."""
        u, v = aux_state.cos_lat_u
        sec2 = torch.tensor(
            self.grid.sec2_lat, dtype=u.dtype, device=u.device,
        )
        ke_nodal = 0.5 * (u**2 + v**2) * sec2
        ke_modal = self.grid.to_modal(ke_nodal)
        return -self.grid.laplacian(ke_modal)

    def orography_tendency(self, device: torch.device | None = None) -> torch.Tensor:
        """−g·∇²(orography) in modal space."""
        oro = self.orography
        # PrimitiveEquationsSigma is not an nn.Module, so model.to(device)
        # does not propagate.  Lazily move + cache orography on first call.
        if device is not None and oro.device != device:
            self.orography = oro = oro.to(device)
        return -self.g * self.grid.laplacian(oro)

    def curl_and_div_tendencies(
        self,
        aux_state: DiagnosticState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Curl and div tendencies from advection + Coriolis + vertical + RT'∇lnPs.

        Aligned with dinosaur reference:
          dζ/dt = -k·∇×((ζ+f)(k×v) + σ̇·∂v/∂σ + RT'∇lnPs)
          dδ/dt = -∇·((ζ+f)(k×v) + σ̇·∂v/∂σ + RT'∇lnPs)

        All three terms are combined in nodal space before a single
        curl_cos_lat / div_cos_lat call, matching the reference exactly.
        """
        grid = self.grid
        u, v = aux_state.cos_lat_u
        vort = aux_state.vorticity
        sec2 = torch.tensor(grid.sec2_lat, dtype=u.dtype, device=u.device)

        # Coriolis parameter
        coriolis = torch.tensor(
            self._coriolis, dtype=u.dtype, device=u.device,
        )
        total_vorticity = vort + coriolis

        # (ζ+f)×(k×v): note k×v = (-v, u), so components of (ζ+f)(k×v) are:
        #   component along lon: -v * (ζ+f)
        #   component along lat:  u * (ζ+f)
        # These are already multiplied by cos(lat) since u,v are cos_lat_u.
        nodal_vorticity_u = -v * total_vorticity * sec2
        nodal_vorticity_v = u * total_vorticity * sec2

        # Vertical advection of velocity: σ̇·∂v/∂σ
        # Reference uses: -σ̇ * u, -σ̇ * v (vertical tendency of cos_lat*velocity)
        if self.include_vertical_advection and aux_state.sigma_dot_full.shape[-3] > 0:
            sigma_dot_u = -sigma_mod.centered_vertical_advection(
                aux_state.sigma_dot_full, u, self.vertical,
            )
            sigma_dot_v = -sigma_mod.centered_vertical_advection(
                aux_state.sigma_dot_full, v, self.vertical,
            )
        else:
            sigma_dot_u = torch.zeros_like(u)
            sigma_dot_v = torch.zeros_like(v)

        # RT'∇lnPs term (with moist virtual-temperature adjustment)
        T_var = aux_state.temperature_variation
        grad_lnps_u, grad_lnps_v = aux_state.cos_lat_grad_log_sp
        adjustment = self._virtual_temperature_adjustment(aux_state)
        rt = self.R * T_var * adjustment
        # These are sec2-weighted because grad_lnps is cos_lat * grad(lnPs)
        # and we need the actual vector: RT'·∇lnPs / cos²θ * sec²θ = RT'·∇lnPs
        vertical_term_u = (sigma_dot_u + rt * grad_lnps_u) * sec2
        vertical_term_v = (sigma_dot_v + rt * grad_lnps_v) * sec2

        # Combine all terms before spectral transform
        combined_u = grid.to_modal(nodal_vorticity_u + vertical_term_u)
        combined_v = grid.to_modal(nodal_vorticity_v + vertical_term_v)

        # Single curl/div call on the combined flux
        vort_tendency = -grid.curl_cos_lat((combined_u, combined_v), clip=False)
        div_tendency = -grid.div_cos_lat((combined_u, combined_v), clip=False)

        return vort_tendency, div_tendency

    def horizontal_scalar_advection(
        self,
        scalar_nodal: torch.Tensor,
        aux_state: DiagnosticState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Horizontal advection of a scalar field.

        Returns (nodal_component, modal_component) of the tendency:
          nodal: scalar * divergence
          modal: -div_sec_lat(u·scalar, v·scalar)
        """
        u, v = aux_state.cos_lat_u
        sec2 = torch.tensor(
            self.grid.sec2_lat, dtype=u.dtype, device=u.device,
        )
        scalar_times_div = scalar_nodal * aux_state.divergence
        flux_u = self.grid.to_modal(u * scalar_nodal * sec2)
        flux_v = self.grid.to_modal(v * scalar_nodal * sec2)
        modal_tendency = -self.grid.div_cos_lat((flux_u, flux_v), clip=False)
        return scalar_times_div, modal_tendency

    def nodal_temperature_vertical_tendency(
        self,
        aux_state: DiagnosticState,
    ) -> torch.Tensor:
        """Vertical advection of T' and T_ref."""
        if not self.include_vertical_advection:
            return torch.zeros_like(aux_state.temperature_variation)
        # σ̇_full · ∂T'/∂σ
        T_var = aux_state.temperature_variation
        result = sigma_mod.centered_vertical_advection(
            aux_state.sigma_dot_full, T_var, self.vertical,
        )
        # σ̇_explicit · ∂T_ref/∂σ (if T_ref varies with level)
        T_ref = torch.tensor(
            self.T_ref, dtype=T_var.dtype, device=T_var.device,
        ).reshape(-1, 1, 1).expand_as(T_var)
        result = result + sigma_mod.centered_vertical_advection(
            aux_state.sigma_dot_explicit, T_ref, self.vertical,
        )
        return result

    def _t_omega_over_sigma_sp(
        self,
        temperature_field: torch.Tensor,
        g_term: torch.Tensor,
        v_dot_grad_log_sp: torch.Tensor,
    ) -> torch.Tensor:
        """Compute nodal T·ω/p using Durran §8.6.3 eq. 8.124 discretization.

        ω/p[n] = v·∇(lnPs)[n] - (1/Δσ[n]) * (α[n]*∑(G[:n]*Δσ[:n])
                                              + α[n-1]*∑(G[:n-1]*Δσ[:n-1]))

        This matches dinosaur _t_omega_over_sigma_sp exactly.

        Args:
            temperature_field: T (nodal) — could be T_ref or T'.
            g_term: G = u·∇lnPs (for mean part) or D+u·∇lnPs (for variation).
            v_dot_grad_log_sp: u·∇(lnPs) in nodal space.

        Returns:
            T · ω/p nodal tendency.
        """
        # cumulative_sigma_integral: ∑(G[i]*Δσ[i]) from top (downward)
        f = sigma_mod.cumulative_sigma_integral(
            g_term, self.vertical, axis=-3, downward=True,
        )
        alpha = torch.tensor(
            get_sigma_ratios(self.vertical),
            dtype=f.dtype, device=f.device,
        ).reshape(-1, 1, 1)
        dsigma = torch.tensor(
            self.vertical.layer_thickness,
            dtype=f.dtype, device=f.device,
        ).reshape(-1, 1, 1)

        # g_part = (α[n]*f[n] + α[n-1]*f[n-1]) / Δσ[n]
        # with α[-1]*f[-1] = 0 (no integral above the top level)
        # Must pad (alpha * f) BEFORE shifting, so shifted term uses α[n-1].
        # JAX: jnp.pad(alpha * f, [(1,0),(0,0),(0,0)])[:-1]
        alpha_f = alpha * f
        alpha_f_padded = torch.cat([
            torch.zeros_like(alpha_f.narrow(-3, 0, 1)),
            alpha_f,
        ], dim=-3)
        # alpha_f_padded[:-1] = [0, α[0]*f[0], ..., α[N-2]*f[N-2]]
        alpha_f_shifted = alpha_f_padded.narrow(-3, 0, f.shape[-3])
        g_part = (alpha * f + alpha_f_shifted) / dsigma

        return temperature_field * (v_dot_grad_log_sp - g_part)

    def nodal_temperature_adiabatic_tendency(
        self,
        aux_state: DiagnosticState,
    ) -> torch.Tensor:
        """κ·T·ω/p adiabatic heating using Durran §8.6.3 discretization.

        Separates into mean-T and variation-T parts as in dinosaur reference:
          mean_t_part = _t_omega_over_sigma_sp(T_ref, g_explicit, v·∇lnPs)
          variation_t_part = _t_omega_over_sigma_sp(T', g_full, v·∇lnPs)
          adiabatic = κ * (mean_t_part + variation_t_part)
        """
        g_explicit = aux_state.u_dot_grad_log_sp
        g_full = g_explicit + aux_state.divergence

        T_ref = torch.tensor(
            self.T_ref, dtype=aux_state.temperature_variation.dtype,
            device=aux_state.temperature_variation.device,
        ).reshape(-1, 1, 1)

        mean_t_part = self._t_omega_over_sigma_sp(
            T_ref, g_explicit, aux_state.u_dot_grad_log_sp,
        )

        # Moist adiabatic correction matches JAX PrimitiveEquationsSigma:
        # variation_component = T' * ((1 + (R_v/R_d - 1) q) / (1 + (Cp_v/Cp_d - 1) q))
        # humidity_ref_component = T_ref * (((R_v/R_d - Cp_v/Cp_d) q) / (1 + (Cp_v/Cp_d - 1) q))
        # variation_and_Tv_part = _t_omega_over_sigma_sp(variation_component + humidity_ref_component, ...)
        q_nodal = self._get_specific_humidity(aux_state)
        if q_nodal is None:
            variation_input = aux_state.temperature_variation
        else:
            gas_const_ratio = self.physics_specs.R_vapor / self.R
            heat_capacity_ratio = self.physics_specs.Cp_vapor / self.physics_specs.Cp
            denom = 1.0 + (heat_capacity_ratio - 1.0) * q_nodal
            variation_component = aux_state.temperature_variation * (
                (1.0 + (gas_const_ratio - 1.0) * q_nodal) / denom
            )
            humidity_reference_component = T_ref * (
                ((gas_const_ratio - heat_capacity_ratio) * q_nodal) / denom
            )
            variation_input = variation_component + humidity_reference_component

        variation_t_part = self._t_omega_over_sigma_sp(
            variation_input, g_full, aux_state.u_dot_grad_log_sp,
        )

        return self.kappa * (mean_t_part + variation_t_part)

    def nodal_log_pressure_tendency(
        self,
        aux_state: DiagnosticState,
    ) -> torch.Tensor:
        """−∑(G_i · Δσ_i) where G = u·∇lnPs (not D+u·∇lnPs).

        Matches dinosaur reference: g = u_dot_grad_log_sp only.
        """
        g = aux_state.u_dot_grad_log_sp
        return -sigma_mod.sigma_integral(g, self.vertical, axis=-3, keepdims=True)

    # ───────────────────────────────────────────────────────────────────
    # Moisture-induced tendencies (virtual-temperature corrections)
    # ───────────────────────────────────────────────────────────────────

    def _get_specific_humidity(
        self, state_or_aux, *, nodal: bool = False,
    ) -> torch.Tensor | None:
        """Extract specific humidity from state/aux, or return None if dry."""
        if self.humidity_key is None:
            return None
        if isinstance(state_or_aux, (State, DiagnosticState)):
            q = self._get_tracer(state_or_aux, self.humidity_key)
        else:
            return None
        return q

    def divergence_tendency_due_to_humidity(
        self, state: State, aux_state: DiagnosticState,
    ) -> torch.Tensor:
        """Divergence tendency from virtual-temperature moisture correction.

        Accounts for moisture-induced terms:
          1. Δ(R·(Tv − T)·lnPs)   (moisture effect on Ps pressure gradient)
          2. Δ(Φ(Tv) − Φ(T))      (moisture effect on geopotential)

        Returns modal divergence tendency, or zero if dry.
        """
        q_nodal = self._get_specific_humidity(aux_state)
        if q_nodal is None:
            return torch.zeros_like(state.divergence)

        grid = self.grid
        R_diff = self.physics_specs.R_vapor - self.R  # R_v - R_d
        T_ref = torch.tensor(
            self.T_ref, dtype=state.vorticity.dtype,
            device=state.vorticity.device,
        ).reshape(-1, 1, 1)

        # --- Term 1: Δ(R·(Tv-T)·lnPs) = R_diff·T_ref·q·Δ(lnPs) ---
        nodal_laplacian_lsp = grid.to_nodal(
            grid.laplacian(state.log_surface_pressure)
        )
        nodal_laplacian_correction = q_nodal * nodal_laplacian_lsp * T_ref * R_diff

        # --- dot-product term: (R_diff·T_ref) · sec²θ · (∇q·∇lnPs) ---
        q_modal = self._get_specific_humidity(state)
        if q_modal is not None:
            cos_lat_grad_q = grid.cos_lat_grad(q_modal, clip=False)
            nodal_cos_lat_grad_q = (
                grid.to_nodal(cos_lat_grad_q[0]),
                grid.to_nodal(cos_lat_grad_q[1]),
            )
            sec2 = torch.tensor(
                grid.sec2_lat, dtype=state.vorticity.dtype,
                device=state.vorticity.device,
            )
            nodal_dot_term = (
                R_diff * T_ref * sec2 * (
                    nodal_cos_lat_grad_q[0] * aux_state.cos_lat_grad_log_sp[0]
                    + nodal_cos_lat_grad_q[1] * aux_state.cos_lat_grad_log_sp[1]
                )
            )
        else:
            nodal_dot_term = torch.zeros_like(state.divergence[:1])

        # --- Term 2: Δ(Φ(Tv) - Φ(T)) via geopotential difference ---
        # On sigma coordinates, dinosaur's _get_geopotential_diff ignores
        # surface_pressure; get_geopotential_diff_sigma uses temperature only.
        # Do not evaluate exp(ln_ps) here: it is unused and can Inf while the
        # moist geopotential correction itself stays finite (JAX-alignment fix).
        temperature = aux_state.temperature_variation + T_ref
        temperature_diff = q_nodal * temperature * (
            self.physics_specs.R_vapor / self.R - 1.0
        )
        # ``get_geopotential_diff_sigma`` is modal; ``laplacian`` is modal-only.
        # Skip modal→nodal→modal before ``laplacian`` (not identity on TL grids).
        geopotential_diff = get_geopotential_diff_sigma(
            grid.to_modal(temperature_diff), self.vertical, self.R,
        )

        return -grid.laplacian(geopotential_diff) - grid.to_modal(
            nodal_dot_term + nodal_laplacian_correction
        )

    def vorticity_tendency_due_to_humidity(
        self, state: State, aux_state: DiagnosticState,
    ) -> torch.Tensor:
        """Vorticity tendency from virtual-temperature moisture correction.

        Computes the curl of the moisture correction to the pressure gradient:
          ∇×(R_diff·T_ref·sec²θ·(∇lnPs × ∇q))

        Returns modal vorticity tendency, or zero if dry.
        """
        q_modal = self._get_specific_humidity(state)
        if q_modal is None:
            return torch.zeros_like(state.vorticity)

        grid = self.grid
        R_diff = self.physics_specs.R_vapor - self.R
        T_ref = torch.tensor(
            self.T_ref, dtype=state.vorticity.dtype,
            device=state.vorticity.device,
        ).reshape(-1, 1, 1)

        cos_lat_grad_q = grid.cos_lat_grad(q_modal, clip=False)
        nodal_cos_lat_grad_q = (
            grid.to_nodal(cos_lat_grad_q[0]),
            grid.to_nodal(cos_lat_grad_q[1]),
        )
        sec2 = torch.tensor(
            grid.sec2_lat, dtype=state.vorticity.dtype,
            device=state.vorticity.device,
        )
        # Cross product: (∇lnPs)_lon * (∇q)_lat - (∇lnPs)_lat * (∇q)_lon
        nodal_curl_term = (
            R_diff * T_ref * sec2 * (
                aux_state.cos_lat_grad_log_sp[0] * nodal_cos_lat_grad_q[1]
                - aux_state.cos_lat_grad_log_sp[1] * nodal_cos_lat_grad_q[0]
            )
        )
        return grid.to_modal(nodal_curl_term)

    def explicit_terms(self, state: State) -> State:
        """Compute explicit tendency terms for IMEX integration.

        Algorithm:
          1. Compute diagnostic state (nodal fields, sigma_dot, u·∇lnPs)
          2. Curl/div tendencies from vorticity flux + Coriolis
          3. Kinetic energy tendency: −∇²(½|u|²)
          4. Orography tendency: −g·∇²(orography)
          5. Horizontal advection of T' and tracers
          6. Vertical advection of T' and T_ref
          7. Adiabatic heating: κT·ω/p
          8. Surface pressure tendency: −Σ(G·Δσ)
          9. Moisture corrections (if humidity_key is set)
          10. Combine and clip highest wavenumber
        """
        grid = self.grid
        aux = compute_diagnostic_state(state, self.coords)

        # 1. Vorticity/divergence tendencies
        vort_tend, div_tend = self.curl_and_div_tendencies(aux)

        # 2. Kinetic energy tendency
        ke_tend = self.kinetic_energy_tendency(aux)

        # 3. Orography tendency
        oro_tend = self.orography_tendency(device=state.vorticity.device)

        # 4. Divergence = curl_div_tendency + KE + orography
        divergence_tendency = div_tend + ke_tend + oro_tend

        # 5. Temperature horizontal advection
        T_horiz_nodal, T_horiz_modal = self.horizontal_scalar_advection(
            aux.temperature_variation, aux,
        )

        # 6. Temperature vertical tendency
        T_vert = self.nodal_temperature_vertical_tendency(aux)

        # 7. Adiabatic heating
        T_adiab = self.nodal_temperature_adiabatic_tendency(aux)

        # 8. Combine temperature tendency
        temperature_tendency = (
            grid.to_modal(T_horiz_nodal + T_vert + T_adiab)
            + T_horiz_modal
        )

        # 9. Surface pressure tendency
        lnps_tendency = grid.to_modal(self.nodal_log_pressure_tendency(aux))

        # 10. Tracer tendencies
        tracers_tendency = {}
        for key, tracer in aux.tracers.items():
            t_nodal, t_modal = self.horizontal_scalar_advection(tracer, aux)
            if self.include_vertical_advection:
                t_vert = sigma_mod.centered_vertical_advection(
                    aux.sigma_dot_full, tracer, self.vertical,
                )
            else:
                t_vert = torch.zeros_like(tracer)
            tracers_tendency[key] = grid.to_modal(t_nodal + t_vert) + t_modal

        # 9. Moisture corrections (virtual temperature effect)
        if self.humidity_key is not None:
            divergence_tendency = divergence_tendency + self.divergence_tendency_due_to_humidity(state, aux)
            vort_tend = vort_tend + self.vorticity_tendency_due_to_humidity(state, aux)

        # Clip highest wavenumber
        result = State(
            vorticity=grid.clip_wavenumbers(vort_tend),
            divergence=grid.clip_wavenumbers(divergence_tendency),
            temperature_variation=grid.clip_wavenumbers(temperature_tendency),
            log_surface_pressure=grid.clip_wavenumbers(lnps_tendency),
            tracers={k: grid.clip_wavenumbers(v) for k, v in tracers_tendency.items()},
            sim_time=state.sim_time,
        )
        return result

    def implicit_terms(self, state: State) -> State:
        """Compute implicit tendency terms.

        The implicit part handles fast gravity-wave propagation:
          D_implicit = −∇²(Φ + R·T_ref·lnPs)
          T_implicit = −H @ D
          lnPs_implicit = −Δσ^T · D
        """
        grid = self.grid
        vertical = self.vertical

        # Geopotential difference from T'
        geo_diff = get_geopotential_diff_sigma(
            state.temperature_variation, vertical, self.R,
        )
        # R * T_ref * lnPs
        T_ref_t = torch.tensor(
            self.T_ref, dtype=state.log_surface_pressure.dtype,
            device=state.log_surface_pressure.device,
        ).reshape(-1, 1, 1)
        rt_lnps = self.R * T_ref_t * state.log_surface_pressure

        # D_implicit = −∇²(Φ_diff + R·T_ref·lnPs)
        divergence_implicit = -grid.laplacian(geo_diff + rt_lnps)

        # T_implicit = −H @ D
        H = torch.tensor(self._H_matrix, dtype=torch.float64,
                         device=state.divergence.device)
        temperature_implicit = -einsum_highest(
            "gh,...hml->...gml", H, state.divergence,
        )

        # lnPs_implicit = −Δσ^T · D
        dsigma = torch.tensor(
            self._dsigma, dtype=state.divergence.dtype,
            device=state.divergence.device,
        ).reshape(-1, 1, 1)
        lnps_implicit = -(dsigma * state.divergence).sum(dim=-3, keepdim=True)

        return State(
            vorticity=torch.zeros_like(state.vorticity),
            divergence=divergence_implicit,
            temperature_variation=temperature_implicit,
            log_surface_pressure=lnps_implicit,
            tracers={k: torch.zeros_like(v) for k, v in state.tracers.items()},
            sim_time=state.sim_time,
        )

    def implicit_inverse(self, state: State, step_size: float) -> State:
        """Apply `(I - step_size * implicit_terms)^-1`, aligned with JAX split path.

        Key alignment choices:
        1. Build and invert per-wavenumber implicit matrices in float64.
        2. Apply inverse in split form (9 block matvecs), mirroring JAX order.
        3. Cast prognostic updates back to the state tensor dtype (JAX-matched).
        """
        n = self.vertical.layers
        L = self.grid.total_wavenumbers
        eta = float(step_size)

        device = state.vorticity.device

        # Build implicit matrix blocks in float64 to mirror JAX.
        lam = np.asarray(self.grid.laplacian_eigenvalues, dtype=np.float64)  # (L,)
        G = np.asarray(self._G_matrix, dtype=np.float64)  # (n, n)
        H = np.asarray(self._H_matrix, dtype=np.float64)  # (n, n)
        dsigma = np.asarray(self._dsigma, dtype=np.float64)  # (n,)
        T_ref = np.asarray(self.T_ref, dtype=np.float64)  # (n,)
        R = float(self.R)

        eye = np.eye(n, dtype=np.float64)[np.newaxis, ...]  # (1, n, n)
        row0 = np.concatenate(
            [
                np.broadcast_to(eye, (L, n, n)),
                eta * np.einsum("l,jk->ljk", lam, G),
                eta * R * np.einsum("l,j->lj", lam, T_ref)[:, :, np.newaxis],
            ],
            axis=2,
        )
        row1 = np.concatenate(
            [
                eta * np.broadcast_to(H[np.newaxis, ...], (L, n, n)),
                np.broadcast_to(eye, (L, n, n)),
                np.zeros((L, n, 1), dtype=np.float64),
            ],
            axis=2,
        )
        row2 = np.concatenate(
            [
                np.broadcast_to((eta * dsigma)[np.newaxis, np.newaxis, :], (L, 1, n)),
                np.zeros((L, 1, n), dtype=np.float64),
                np.ones((L, 1, 1), dtype=np.float64),
            ],
            axis=2,
        )
        implicit_matrix = np.concatenate([row0, row1, row2], axis=1)  # (L, 2n+1, 2n+1)
        inverse = np.linalg.inv(implicit_matrix)

        inv_t = torch.tensor(inverse, dtype=torch.float64, device=device)

        def _vertical_matvec_per_wavenumber(mat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            # mat: (L, out_n, in_n), x: (in_n, m, L) -> (out_n, m, L)
            x_l = x.permute(2, 0, 1)  # (L, in_n, m)
            out_l = torch.einsum("loi,lim->lom", mat, x_l)
            return out_l.permute(1, 2, 0)

        div = state.divergence.to(torch.float64)
        temp = state.temperature_variation.to(torch.float64)
        logp = state.log_surface_pressure.to(torch.float64)

        div_slice = slice(0, n)
        temp_slice = slice(n, 2 * n)
        logp_slice = slice(2 * n, 2 * n + 1)

        div_out = (
            _vertical_matvec_per_wavenumber(inv_t[:, div_slice, div_slice], div)
            + _vertical_matvec_per_wavenumber(inv_t[:, div_slice, temp_slice], temp)
            + _vertical_matvec_per_wavenumber(inv_t[:, div_slice, logp_slice], logp)
        )
        temp_out = (
            _vertical_matvec_per_wavenumber(inv_t[:, temp_slice, div_slice], div)
            + _vertical_matvec_per_wavenumber(inv_t[:, temp_slice, temp_slice], temp)
            + _vertical_matvec_per_wavenumber(inv_t[:, temp_slice, logp_slice], logp)
        )
        logp_out = (
            _vertical_matvec_per_wavenumber(inv_t[:, logp_slice, div_slice], div)
            + _vertical_matvec_per_wavenumber(inv_t[:, logp_slice, temp_slice], temp)
            + _vertical_matvec_per_wavenumber(inv_t[:, logp_slice, logp_slice], logp)
        )

        # Internal matvecs use float64; cast back to state dtype so IMEX stages
        # stay dtype-consistent with tracers and tendencies (matches JAX).
        out_dtype = state.vorticity.dtype
        return State(
            vorticity=state.vorticity,
            divergence=div_out.to(out_dtype),
            temperature_variation=temp_out.to(out_dtype),
            log_surface_pressure=logp_out.to(out_dtype),
            tracers=state.tracers,
            sim_time=state.sim_time,
        )

    def as_imex_ode(self) -> time_integration.ImplicitExplicitODE:
        """Return an ImplicitExplicitODE interface wrapping this class."""
        return time_integration.ImplicitExplicitODE.from_functions(
            explicit_terms=self.explicit_terms,
            implicit_terms=self.implicit_terms,
            implicit_inverse=self.implicit_inverse,
        )


# Aliases for convenience
PrimitiveEquations = PrimitiveEquationsSigma
