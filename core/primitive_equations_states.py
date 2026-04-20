# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""Primitive equation state construction helpers — PyTorch implementation.

Provides initial conditions for standard test cases:
  - isothermal_rest_atmosphere (Held-Suarez)
  - steady_state_jw (Jablonowski-Williamson baroclinic wave)
  - baroclinic_perturbation_jw
  - gaussian_scalar
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import torch

from tornado_gcm.core import coordinate_systems
from tornado_gcm.core import primitive_equations
from tornado_gcm.core import spherical_harmonic

# Keys for aux_features dict
OROGRAPHY = "orography"
REF_TEMP_KEY = "reference_temperature"
GEOPOTENTIAL = "geopotential"


def isothermal_rest_atmosphere(
    coords: coordinate_systems.CoordinateSystem,
    physics_specs,
    tref: float = 288.0,
    p0: float = 1e5,
    p1: float = 0.0,
    surface_height: Optional[torch.Tensor] = None,
    device: str | torch.device = "cpu",
) -> tuple[Callable, dict[str, Any]]:
    """Create an isothermal rest atmosphere (Held-Suarez test case).

    Args:
        coords: coordinate system.
        physics_specs: SimUnits instance.
        tref: reference temperature (K, nondimensional via physics_specs).
        p0: mean surface pressure.
        p1: perturbation amplitude for surface pressure.
        surface_height: optional orography (nodal, nondimensional).

    Returns:
        (initial_state_fn, aux_features) tuple.
    """
    grid = coords.horizontal
    n_layers = coords.vertical.layers

    if surface_height is not None:
        orography = spherical_harmonic.truncated_modal_orography(
            surface_height, coords, wavenumbers_to_clip=1
        ) if hasattr(spherical_harmonic, 'truncated_modal_orography') else (
            primitive_equations.truncated_modal_orography(surface_height, coords)
        )
    else:
        orography = torch.zeros(grid.modal_shape, device=device)

    ref_temp = np.full(n_layers, tref, dtype=np.float64)

    def initial_state_fn(rng_key=None):
        modal_shape = coords.modal_shape
        surface_modal_shape = coords.surface_modal_shape

        vorticity = torch.zeros(modal_shape, device=device)
        divergence = torch.zeros(modal_shape, device=device)
        temperature_variation = torch.zeros(modal_shape, device=device)

        # Surface pressure: constant p0 (with optional small perturbation)
        log_ps_nodal = torch.full(
            grid.nodal_shape, np.log(p0), device=device, dtype=torch.float32,
        )
        if p1 > 0 and rng_key is not None:
            lon, sin_lat = grid.nodal_mesh
            lon_t = torch.tensor(lon, dtype=torch.float32, device=device)
            lat_t = torch.tensor(np.arcsin(sin_lat), dtype=torch.float32, device=device)
            perturbation = p1 * torch.sin(5 * lon_t) * torch.exp(
                -10 * lat_t**2
            )
            log_ps_nodal = torch.log(
                torch.exp(log_ps_nodal) + perturbation
            )

        log_surface_pressure = grid.to_modal(log_ps_nodal)
        # Truncate to surface modal shape
        if log_surface_pressure.shape != surface_modal_shape:
            log_surface_pressure = log_surface_pressure.unsqueeze(0)
            log_surface_pressure = log_surface_pressure[:surface_modal_shape[0]]

        return primitive_equations.State(
            vorticity=vorticity,
            divergence=divergence,
            temperature_variation=temperature_variation,
            log_surface_pressure=log_surface_pressure,
        )

    aux_features = {
        OROGRAPHY: orography,
        REF_TEMP_KEY: ref_temp,
    }
    return initial_state_fn, aux_features


def steady_state_jw(
    coords: coordinate_systems.CoordinateSystem,
    physics_specs,
    u0: float = 35.0,
    p0: float = 1e5,
    t0: float = 288.0,
    delta_t: float = 4.8e5,
    gamma: float = 0.005,
    eta_tropo: float = 0.2,
    eta0: float = 0.252,
    device: str | torch.device = "cpu",
) -> tuple[Callable, dict[str, Any]]:
    """Jablonowski-Williamson baroclinic wave steady state.

    Args:
        coords: coordinate system.
        physics_specs: SimUnits instance.
        u0: max wind speed (nondimensional).
        p0: reference surface pressure.
        t0: reference temperature.
        delta_t: temperature parameter.
        gamma: lapse rate.
        eta_tropo: tropopause eta.
        eta0: reference eta.

    Returns:
        (initial_state_fn, aux_features) tuple.
    """
    grid = coords.horizontal
    vertical = coords.vertical
    R = physics_specs.R
    g = physics_specs.g
    Omega = physics_specs.angular_velocity
    a = physics_specs.radius
    kappa = physics_specs.kappa
    n_layers = vertical.layers

    eta = vertical.centers
    _, sin_lat_np = grid.nodal_axes
    cos_lat_np = np.sqrt(1 - sin_lat_np**2)
    lon_np, _ = grid.nodal_mesh
    lat_np = np.arcsin(sin_lat_np)

    def _eta_v(e):
        return (e - eta0) * np.pi / 2.0

    def _get_reference_temperature(e):
        """Reference temperature T(eta)."""
        ev = _eta_v(e)
        T = t0 * e ** (R * gamma / g)
        if e > eta_tropo:
            T += delta_t * (eta_tropo - e) ** 5
        return T

    ref_temp = np.array([_get_reference_temperature(e) for e in eta], dtype=np.float64)

    def _get_zonal_wind(lat, e):
        """Zonal wind u(lat, eta)."""
        ev = _eta_v(e)
        return u0 * np.cos(ev) ** 1.5 * np.sin(2 * lat) ** 2

    def _get_vorticity(lat, e):
        """Analytic vorticity from zonal wind."""
        ev = _eta_v(e)
        return (
            -4 * u0 / a * np.cos(ev) ** 1.5
            * np.sin(lat) * np.cos(lat)
            * (2 - 5 * np.sin(lat) ** 2)
        )

    def initial_state_fn(rng_key=None):
        modal_shape = coords.modal_shape
        surface_modal_shape = coords.surface_modal_shape

        # Vorticity from analytic formula
        vort_nodal = np.zeros((n_layers,) + grid.nodal_shape)
        for k in range(n_layers):
            for j in range(grid.latitude_nodes):
                vort_nodal[k, :, j] = _get_vorticity(lat_np[0, j], eta[k])

        vort_t = torch.tensor(vort_nodal, dtype=torch.float32, device=device)
        vorticity = grid.to_modal(vort_t)

        divergence = torch.zeros(modal_shape, device=device)

        # Temperature variation from reference
        T_nodal = np.zeros((n_layers,) + grid.nodal_shape)
        for k in range(n_layers):
            T_nodal[k] = ref_temp[k]
        temperature_variation = torch.zeros(modal_shape, device=device)

        # Surface pressure = p0
        log_ps_nodal = torch.full(
            grid.nodal_shape, np.log(p0), dtype=torch.float32, device=device,
        )
        log_surface_pressure = grid.to_modal(log_ps_nodal).unsqueeze(0)
        if log_surface_pressure.shape[0] > surface_modal_shape[0]:
            log_surface_pressure = log_surface_pressure[:surface_modal_shape[0]]

        return primitive_equations.State(
            vorticity=vorticity,
            divergence=divergence,
            temperature_variation=temperature_variation,
            log_surface_pressure=log_surface_pressure,
        )

    # Orography = 0 for this test case
    orography = torch.zeros(grid.modal_shape, device=device)

    aux_features = {
        OROGRAPHY: orography,
        REF_TEMP_KEY: ref_temp,
    }
    return initial_state_fn, aux_features


def baroclinic_perturbation_jw(
    coords: coordinate_systems.CoordinateSystem,
    physics_specs,
    u_perturb: float = 1.0,
    lon_location: float = np.pi / 9,
    lat_location: float = 2 * np.pi / 9,
    perturbation_radius: float = 0.1,
    device: str | torch.device = "cpu",
) -> primitive_equations.State:
    """Perturbation triggering baroclinic instability (JW test).

    Returns a State containing the perturbation (to add to steady state).
    """
    grid = coords.horizontal
    a = physics_specs.radius
    n_layers = coords.vertical.layers

    lon_np, sinlat_np = grid.nodal_mesh
    lat_np = np.arcsin(sinlat_np)

    # Great-circle distance from perturbation center
    r = np.arccos(
        np.sin(lat_location) * np.sin(lat_np)
        + np.cos(lat_location) * np.cos(lat_np) * np.cos(lon_np - lon_location)
    )

    # Gaussian envelope
    envelope = np.exp(-(r / perturbation_radius) ** 2)

    # Vorticity perturbation (same for all levels)
    vort_pert_nodal = np.zeros((n_layers,) + grid.nodal_shape)
    for k in range(n_layers):
        vort_pert_nodal[k] = u_perturb / a * envelope

    vort_t = torch.tensor(vort_pert_nodal, dtype=torch.float32, device=device)
    vorticity = grid.to_modal(vort_t)

    modal_shape = coords.modal_shape
    surface_modal_shape = coords.surface_modal_shape

    return primitive_equations.State(
        vorticity=vorticity,
        divergence=torch.zeros(modal_shape, device=device),
        temperature_variation=torch.zeros(modal_shape, device=device),
        log_surface_pressure=torch.zeros(surface_modal_shape, device=device),
    )


def gaussian_scalar(
    coords: coordinate_systems.CoordinateSystem,
    lon_location: float = np.pi / 9,
    lat_location: float = 2 * np.pi / 9,
    perturbation_radius: float = 0.2,
    amplitude: float = 1.0,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Gaussian scalar field in modal representation."""
    grid = coords.horizontal
    lon_np, sinlat_np = grid.nodal_mesh
    lat_np = np.arcsin(sinlat_np)

    r = np.arccos(
        np.sin(lat_location) * np.sin(lat_np)
        + np.cos(lat_location) * np.cos(lat_np) * np.cos(lon_np - lon_location)
    )
    nodal = amplitude * np.exp(-(r / perturbation_radius) ** 2)
    return grid.to_modal(torch.tensor(nodal, dtype=torch.float32, device=device))
