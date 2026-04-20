"""Physics diagnostics — energy, mass, precipitation extractors.

Provides:
  - Trajectory-level diagnostics (energy, mass, P-E) for
    PhysicsConstrainedLoss.
  - Step-level DiagnosticFn compatible with the ModelState pipeline
    (matching the legacy ``diagnostics_fn(model_state, tendencies, forcing)``
    protocol).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

import torch

from pytorch_src.core import primitive_equations
from pytorch_src.neural.fixers import compute_total_energy, compute_dry_air_mass


def extract_energies(
    trajectory: list[primitive_equations.State],
    initial_state: primitive_equations.State,
) -> torch.Tensor:
    """Extract total energy at each saved step (including initial).

    Returns: (K+1,) tensor of energy values.
    """
    energies = [compute_total_energy(initial_state)]
    for state in trajectory:
        energies.append(compute_total_energy(state))
    return torch.stack(energies)


def extract_masses(
    trajectory: list[primitive_equations.State],
    initial_state: primitive_equations.State,
) -> torch.Tensor:
    """Extract global dry-air mass at each saved step (including initial).

    Returns: (K+1,) tensor of mass proxies.
    """
    masses = [compute_dry_air_mass(initial_state)]
    for state in trajectory:
        masses.append(compute_dry_air_mass(state))
    return torch.stack(masses)


def extract_precipitation_evaporation(
    trajectory: list[primitive_equations.State],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract precipitation and evaporation from tracer fields.

    Convention:
      - 'specific_humidity' tracer: decrease → precipitation, increase → evaporation
      - precipitation >= 0, evaporation <= 0

    Returns: (precipitation, evaporation) each shape (K,).
    """
    precip_list = []
    evap_list = []

    for i, state in enumerate(trajectory):
        q = state.tracers.get("specific_humidity")
        if q is None:
            precip_list.append(torch.tensor(0.0))
            evap_list.append(torch.tensor(0.0))
            continue
        # Global mean of q change → proxy for P-E
        q_mean = q[..., 0, 0].sum()
        # Negative q tendency → precipitation (drying)
        precip = torch.clamp(-q_mean, min=0.0)
        evap = torch.clamp(q_mean, max=0.0)
        precip_list.append(precip)
        evap_list.append(evap)

    return (
        torch.stack(precip_list),
        torch.stack(evap_list),
    )


def compute_trajectory_diagnostics(
    trajectory: list[primitive_equations.State],
    initial_state: primitive_equations.State,
) -> dict[str, torch.Tensor]:
    """Compute all diagnostic quantities needed for PhysicsConstrainedLoss.

    Returns dict with keys: 'energies', 'masses', 'precipitation', 'evaporation'.
    """
    energies = extract_energies(trajectory, initial_state)
    masses = extract_masses(trajectory, initial_state)
    precip, evap = extract_precipitation_evaporation(trajectory)

    return {
        "energies": energies,
        "masses": masses,
        "precipitation": precip,
        "evaporation": evap,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Step-level diagnostics  (ModelState pipeline)
# ═══════════════════════════════════════════════════════════════════════════

def precipitation_rate_diagnostic(
    state: primitive_equations.State,
    previous_state: Optional[primitive_equations.State] = None,
) -> torch.Tensor:
    """Compute precipitation rate from humidity tendency.

    P - E ~ -(dq/dt) integrated over levels.  When ``previous_state`` is
    provided (i.e. *memory*), the tendency is estimated as a finite
    difference.

    Returns:
        Scalar tensor (global mean P-E rate), or zero if no humidity tracer.
    """
    q_now = state.tracers.get("specific_humidity")
    if q_now is None:
        return torch.tensor(0.0, device=state.vorticity.device)
    if previous_state is not None:
        q_prev = previous_state.tracers.get("specific_humidity", q_now)
        dq = q_now - q_prev
    else:
        dq = q_now
    # Sum over levels, global mean (m=0, l=0 coefficient)
    return -dq[..., 0, 0].sum()


def heating_rate_diagnostic(
    state: primitive_equations.State,
    previous_state: Optional[primitive_equations.State] = None,
) -> torch.Tensor:
    """Compute column-integrated heating rate from temperature tendency.

    Returns scalar tensor (global mean heating rate proxy).
    """
    t_now = state.temperature_variation
    if previous_state is not None:
        t_prev = previous_state.temperature_variation
        dt = t_now - t_prev
    else:
        dt = t_now
    return dt[..., 0, 0].sum()


class CombinedStepDiagnostics:
    """Step-level diagnostics function compatible with the ModelState pipeline.

    Implements the legacy ``DiagnosticFn`` protocol::

        diagnostics_fn(model_state, physics_tendencies, forcing) → dict

    Aggregates precipitation rate, heating rate, and energy.
    """

    def __call__(
        self,
        model_state: Any,
        physics_tendencies: Any = None,
        forcing: Any = None,
    ) -> dict[str, torch.Tensor]:
        state = model_state.state
        memory = model_state.memory

        diags: dict[str, torch.Tensor] = {}

        # Precipitation / evaporation rate
        diags["P_minus_E_rate"] = precipitation_rate_diagnostic(state, memory)

        # Column heating rate
        diags["heating_rate"] = heating_rate_diagnostic(state, memory)

        # Instantaneous energy and mass
        diags["total_energy"] = compute_total_energy(state)
        diags["dry_air_mass"] = compute_dry_air_mass(state)

        return diags
