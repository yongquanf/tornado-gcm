"""Conservation fixers — Z2 (FP64) precision zone.

Implements:
  - KahanAccumulator: compensated summation for FP64 integrals
  - GlobalDryAirMassFixer: adjusts surface pressure to conserve dry air mass
  - GlobalEnergyFixer: rescales temperature to conserve total energy
  - F64ConservationFixer: combined fixer (mass + energy) in FP64
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pytorch_src.core import primitive_equations
from pytorch_src.precision.zone_cast import f64_math


# ═══════════════════════════════════════════════════════════════════════════
# Kahan compensated summation
# ═══════════════════════════════════════════════════════════════════════════

class KahanAccumulator:
    """Kahan compensated summation for reducing round-off in FP64 integrals.

    Usage::

        acc = KahanAccumulator()
        for x in values:
            acc.add(x)
        total = acc.sum
    """

    def __init__(self):
        self.sum: torch.Tensor | None = None
        self.compensation: torch.Tensor | None = None

    def add(self, x: torch.Tensor) -> None:
        if self.sum is None:
            self.sum = x.clone().to(torch.float64)
            self.compensation = torch.zeros_like(self.sum)
            return
        x = x.to(torch.float64)
        y = x - self.compensation
        t = self.sum + y
        self.compensation = (t - self.sum) - y
        self.sum = t

    def reset(self) -> None:
        self.sum = None
        self.compensation = None


def kahan_sum(tensors: list[torch.Tensor], dim: int | None = None) -> torch.Tensor:
    """Kahan-compensated summation over a list of tensors.

    If dim is given, each tensor is summed along that dim first, then
    results are accumulated with Kahan compensation.
    """
    acc = KahanAccumulator()
    for t in tensors:
        val = t.sum(dim=dim) if dim is not None else t
        acc.add(val)
    assert acc.sum is not None
    return acc.sum


# ═══════════════════════════════════════════════════════════════════════════
# Global integrals (FP64)
# ═══════════════════════════════════════════════════════════════════════════

@f64_math
def _global_mean_modal(x: torch.Tensor) -> torch.Tensor:
    """Global mean of a modal field via (0,0) coefficient.

    For a properly normalized SH expansion, the (0,0) mode is the global
    mean times a normalization constant.  For simplicity we take the
    (m=0, l=0) entry as the proxy.
    """
    # x shape: (..., M, L) — the (0, 0) entry is at [..., 0, 0]
    return x[..., 0, 0]


def compute_dry_air_mass(state: primitive_equations.State) -> torch.Tensor:
    """Approximate global dry-air mass from modal log_surface_pressure.

    Uses the (0,0) spectral mode as a proxy for the global mean.
    """
    return _global_mean_modal(state.log_surface_pressure)


def compute_total_energy(state: primitive_equations.State) -> torch.Tensor:
    """Approximate total energy from modal state.

    Energy ~ sum of (vorticity² + divergence² + temperature²) spectral norms.
    This is not the full atmospheric energy integral but a reasonable proxy
    for conservation monitoring / correction.

    Computes in the input's dtype — caller (F64ConservationFixer) is
    responsible for ensuring inputs are in the desired precision.
    """
    ke = (state.vorticity ** 2 + state.divergence ** 2).sum()
    te = (state.temperature_variation ** 2).sum()
    return ke + te


# ═══════════════════════════════════════════════════════════════════════════
# Fixers
# ═══════════════════════════════════════════════════════════════════════════

class GlobalDryAirMassFixer(nn.Module):
    """Correct log_surface_pressure to conserve dry air mass.

    After a time step, the global mean of log(Ps) may drift.
    This fixer adds a uniform correction to the (0,0) mode to restore
    the original global mean.
    """

    def forward(
        self,
        state: primitive_equations.State,
        state_before: primitive_equations.State,
    ) -> primitive_equations.State:
        # Compute drift in (0,0) mode — use input's dtype (set by
        # F64ConservationFixer to the workspace precision).
        lsp = state.log_surface_pressure
        lsp_ref = state_before.log_surface_pressure
        drift = lsp[..., 0, 0] - lsp_ref[..., 0, 0]

        # Correct: subtract drift from (0,0)
        lsp_new = lsp.clone()
        lsp_new[..., 0, 0] = lsp_new[..., 0, 0] - drift

        return primitive_equations.State(
            vorticity=state.vorticity,
            divergence=state.divergence,
            temperature_variation=state.temperature_variation,
            log_surface_pressure=lsp_new,
            tracers=state.tracers,
            sim_time=state.sim_time,
        )


class GlobalEnergyFixer(nn.Module):
    """Conserve total energy via additive temperature correction.

    Following the original NeuralGCM design: compute the energy deficit
    after a time step and absorb it as a uniform (global-mean) temperature
    offset applied to the (0,0) spectral mode.

    Additive correction is numerically stable — avoids the compounding
    oscillations that multiplicative rescaling produces.
    """

    def forward(
        self,
        state: primitive_equations.State,
        state_before: primitive_equations.State,
    ) -> primitive_equations.State:
        # Compute energy deficit (should be zero for perfect conservation)
        e_before = compute_total_energy(state_before)
        e_after = compute_total_energy(state)
        e_deficit = e_before - e_after  # positive means energy was lost

        # Correct via (0,0) temperature mode:
        #   E_T = sum(T^2),  dE_T/d(T[0,0]) = 2*T[0,0]
        #   => delta = e_deficit / (2 * T[0,0])         (linearised)
        #   Exact:  (T[0,0]+d)^2 - T[0,0]^2 = e_deficit
        #           d = -T[0,0] + sign(T[0,0]) * sqrt(T[0,0]^2 + e_deficit)
        T_00 = state.temperature_variation[..., 0, 0]
        discriminant = T_00 ** 2 + e_deficit
        if discriminant.min() <= 0:
            # Cannot fix — would require imaginary correction
            return state

        delta = -T_00 + torch.sign(T_00) * torch.sqrt(discriminant)

        temp_fixed = state.temperature_variation.clone()
        temp_fixed[..., 0, 0] = temp_fixed[..., 0, 0] + delta

        return primitive_equations.State(
            vorticity=state.vorticity,
            divergence=state.divergence,
            temperature_variation=temp_fixed,
            log_surface_pressure=state.log_surface_pressure,
            tracers=state.tracers,
            sim_time=state.sim_time,
        )


class F64ConservationFixer(nn.Module):
    """Combined conservation fixer: dry-air mass + total energy.

    Applies both fixes sequentially in Z2 (FP64) precision:
      1. Mass fix (adjust log_surface_pressure)
      2. Energy fix (rescale temperature)

    This is the standard fixer used in NeuralGCMModel.
    """

    def __init__(self, fix_mass: bool = True, fix_energy: bool = True, dtype: torch.dtype = torch.float64):
        super().__init__()
        self.dtype = dtype  # informational; actual precision is set by Z2 policy via model
        self.mass_fixer = GlobalDryAirMassFixer() if fix_mass else None
        self.energy_fixer = GlobalEnergyFixer() if fix_energy else None

    def forward(
        self,
        state: primitive_equations.State,
        state_before: primitive_equations.State,
    ) -> primitive_equations.State:
        # Inputs arrive pre-cast to Z2 policy dtype by the model's
        # _apply_conservation_fix (PZHA zone boundary).  Sub-fixers
        # compute in the input's dtype — no re-casting here.
        if self.mass_fixer is not None:
            state = self.mass_fixer(state, state_before)
        if self.energy_fixer is not None:
            state = self.energy_fixer(state, state_before)
        return state


# ═══════════════════════════════════════════════════════════════════════════
# Z2-Lite: fast conservation fixer (no full-tensor upcast)
# ═══════════════════════════════════════════════════════════════════════════

def compute_total_energy_lite(state: primitive_equations.State) -> torch.Tensor:
    """Energy proxy with FP32 multiply + FP64 accumulation.

    Uses ``sum(dtype=torch.float64)`` to accumulate FP32 squares into an
    FP64 scalar *without* allocating a full FP64 copy of the tensor.
    This eliminates the memory-bandwidth bottleneck of full-state upcast
    while retaining ~7 digits of precision per element (limited by FP32
    squaring) and exact accumulation in FP64.

    Precision analysis (2.8°, 64×32 grid, N≈2048 coefficients):
      - Per-element relative error from FP32 squaring: ε ≈ 2⁻²⁴ ≈ 6e-8
      - Accumulated error (FP64 sum): √N × ε ≈ 45 × 6e-8 ≈ 3e-6 (relative)
      - Energy deficit between steps is O(1e-3..1e-1) → measurable with margin
      - Vs full FP64: drift ~5e-8;  vs full FP32: drift ~1e-6
      - Expected drift with lite: ~1e-7 (10× better than FP32, 2× worse than FP64)
    """
    ke = (state.vorticity ** 2 + state.divergence ** 2).sum(
        dtype=torch.float64
    )
    te = (state.temperature_variation ** 2).sum(dtype=torch.float64)
    return ke + te


class LiteConservationFixer(nn.Module):
    """Z2-Lite: fast conservation fixer without full-tensor upcast.

    Design: operates directly on FP32 state tensors, using pinpoint FP64
    only where needed (scalar arithmetic + FP64-accumulated reductions).

    Performance model (vs F64ConservationFixer):
      - Eliminates: 2× full-state upcast (FP32→FP64) + 1× full-state downcast
      - Retains:    3 sum-reductions (FP64 accum) + scalar FP64 arithmetic
      - Expected Z2 time: ~40ms → ~2-3ms at 2.8° (T21)
      - Net PZHA overhead: ~33% → ~5-8% vs FP32 baseline

    Conservation precision:
      - Mass: identical to F64 fixer (scalar (0,0) mode, exact)
      - Energy: ~1e-7 drift/200 steps (vs 5e-8 full-FP64, 1e-6 full-FP32)
    """

    def __init__(self, fix_mass: bool = True, fix_energy: bool = True):
        super().__init__()
        self.fix_mass = fix_mass
        self.fix_energy = fix_energy

    # noinspection PyMethodMayBeStatic
    @property
    def bypass_upcast(self) -> bool:
        """Signal to _apply_conservation_fix to skip full-state upcast."""
        return True

    def forward(
        self,
        state: primitive_equations.State,
        state_before: primitive_equations.State,
    ) -> primitive_equations.State:
        # ── Mass fix: scalar (0,0) mode only ──────────────────────────
        if self.fix_mass:
            lsp = state.log_surface_pressure
            # FP64 scalar arithmetic on (0,0) mode
            drift = (
                lsp[..., 0, 0].to(torch.float64)
                - state_before.log_surface_pressure[..., 0, 0].to(torch.float64)
            )
            lsp_new = lsp.clone()
            lsp_new[..., 0, 0] = lsp_new[..., 0, 0] - drift.to(lsp.dtype)
            state = primitive_equations.State(
                vorticity=state.vorticity,
                divergence=state.divergence,
                temperature_variation=state.temperature_variation,
                log_surface_pressure=lsp_new,
                tracers=state.tracers,
                sim_time=state.sim_time,
            )

        # ── Energy fix: FP32 squares + FP64 accumulation ─────────────
        if self.fix_energy:
            e_before = compute_total_energy_lite(state_before)
            e_after = compute_total_energy_lite(state)
            e_deficit = e_before - e_after  # FP64 scalar

            T_00 = state.temperature_variation[..., 0, 0].to(torch.float64)
            discriminant = T_00 ** 2 + e_deficit
            if discriminant.min() > 0:
                delta = -T_00 + torch.sign(T_00) * torch.sqrt(discriminant)
                temp_fixed = state.temperature_variation.clone()
                temp_fixed[..., 0, 0] = (
                    temp_fixed[..., 0, 0] + delta.to(state.temperature_variation.dtype)
                )
                state = primitive_equations.State(
                    vorticity=state.vorticity,
                    divergence=state.divergence,
                    temperature_variation=temp_fixed,
                    log_surface_pressure=state.log_surface_pressure,
                    tracers=state.tracers,
                    sim_time=state.sim_time,
                )

        return state
