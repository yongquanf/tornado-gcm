# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""Conservation fixers — Z2 (FP64) precision zone.

Implements:
  - KahanAccumulator: compensated summation for FP64 integrals
  - GlobalDryAirMassFixer: adjusts surface pressure to conserve dry air mass
  - GlobalEnergyFixer: rescales temperature to conserve total energy
  - F64ConservationFixer: combined fixer (mass + energy) in FP64
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from tornado_gcm.core import primitive_equations
from tornado_gcm.precision.zone_cast import f64_math


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


def build_physical_ctx(coords, physics_specs, reference_temperature,
                       orography=None, g_nondim=None):
    """Bundle the context needed for the physical (mass-weighted) integrals.

    Returns a dict usable as ``kwargs`` for :func:`compute_total_energy_physical`
    plus the constants the fixers need.  Returns ``None`` if the required pieces
    are missing, so callers can fall back explicitly rather than silently.
    """
    if coords is None or physics_specs is None or reference_temperature is None:
        return None
    surface_geopotential = None
    if orography is not None:
        g = float(g_nondim if g_nondim is not None else physics_specs.g)
        surface_geopotential = coords.horizontal.to_nodal(
            orography.to(torch.float64)) * g
    return {
        "coords": coords,
        "reference_temperature": reference_temperature,
        "physics_specs": physics_specs,
        "surface_geopotential": surface_geopotential,
    }


_UNIT00_NODAL_CACHE: dict = {}


def unit00_nodal(grid) -> float:
    """Nodal value of the (0,0) mode with unit coefficient.

    A uniform nodal offset ``d`` corresponds to a (0,0) *coefficient*
    increment of ``d / unit00_nodal(grid)`` -- ignoring this factor is a
    silent scaling error (it cancels for the legacy proxy metric, which lives
    in coefficient space, but not for the physical integrals).

    The value is a grid constant: memoized per modal shape.  Fixers capture
    it at construction time so their traced forward passes contain no
    data-dependent operations (the in-graph ``float()`` used to break
    ``torch.export`` in zone discovery's DAG path -- ISSUE-016).
    """
    key = tuple(grid.modal_shape)
    cached = _UNIT00_NODAL_CACHE.get(key)
    if cached is not None:
        return cached
    m, l = key
    probe = torch.zeros(1, m, l, dtype=torch.float64)
    probe[..., 0, 0] = 1.0
    val = float(grid.to_nodal(probe).reshape(-1)[0])
    _UNIT00_NODAL_CACHE[key] = val
    return val


def compute_column_mass(state: primitive_equations.State, ctx) -> torch.Tensor:
    """Physical global dry-air mass proxy ∫ Ps/g dA (nondimensional).

    Used by the mass fixer, which needs a *mass* integral rather than the
    (0,0) mode of log(Ps): the two differ whenever the surface pressure field
    is not spatially uniform.
    """
    grid = ctx["coords"].horizontal
    ps = torch.exp(grid.to_nodal(state.log_surface_pressure.to(torch.float64)))
    g = float(ctx["physics_specs"].g)
    return grid.integrate(ps.squeeze(-3) / g)



def compute_total_energy_physical(
    state: primitive_equations.State,
    coords,
    reference_temperature,
    physics_specs,
    surface_geopotential: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mass-weighted dry total-energy integral (physical invariant).

    Unlike ``compute_total_energy`` (a raw spectral-coefficient proxy that mixes
    units and ignores the spherical/level quadrature), this is the physically
    meaningful column-integrated total energy of a hydrostatic primitive-equation
    atmosphere in sigma coordinates::

        E = (1/g) ∫_sphere Ps(λ,φ) ∫_0^1 [ Cp·T + ½(u²+v²) ] dσ dA
            + (1/g) ∫_sphere Ps·Φ_s dA      (surface-geopotential term, optional)

    where ``dp = Ps dσ`` is the hydrostatic mass element and the enthalpy form
    ``Cp·T`` folds in the gravitational potential energy via the hydrostatic
    relation.  The horizontal integral uses the grid's Gaussian quadrature
    (cos-lat weights × radius²); the vertical integral uses the sigma-layer
    thicknesses.  All quantities are in the model's nondimensional units, so
    ``Cp`` and ``g`` come from ``physics_specs``.

    This quantity is an actual conserved invariant of the dynamics, so its
    step-to-step drift is a meaningful conservation diagnostic (the raw proxy
    is not — it collapses ~99.96% on step 1 from the init representation jump).

    Args:
        state: modal prognostic State.
        coords: CoordinateSystem (``.horizontal`` Grid, ``.vertical`` sigma).
        reference_temperature: (n_levels,) T_ref profile; T = T_ref + T'.
        physics_specs: SimUnits providing ``Cp`` and ``g`` (nondimensional).
        surface_geopotential: optional (1, x, y) nodal Φ_s = orography·g.

    Returns:
        Scalar tensor: the total energy (nondimensional units).
    """
    from tornado_gcm.core.spherical_harmonic import vor_div_to_uv_nodal

    grid = coords.horizontal
    sigma = coords.vertical

    vort = state.vorticity.to(torch.float64)
    div = state.divergence.to(torch.float64)
    tvar = state.temperature_variation.to(torch.float64)
    lnps = state.log_surface_pressure.to(torch.float64)

    # Velocities (nodal) from modal vorticity / divergence.
    u, v = vor_div_to_uv_nodal(grid, vort, div)

    # Full temperature (nodal): T = T_ref + T'.
    t_ref = torch.as_tensor(
        np.asarray(reference_temperature), dtype=torch.float64, device=tvar.device
    ).reshape(-1, 1, 1)
    temperature = t_ref + grid.to_nodal(tvar)

    # Surface pressure (nodal): Ps = exp(lnps).
    ps = torch.exp(grid.to_nodal(lnps))  # (..., 1, x, y)

    cp = float(physics_specs.Cp)
    g = float(physics_specs.g)
    d_sigma = torch.as_tensor(
        np.asarray(sigma.layer_thickness), dtype=torch.float64, device=tvar.device
    ).reshape(-1, 1, 1)

    # Column specific energy per unit σ-mass: Cp·T + ½(u²+v²).
    specific_energy = cp * temperature + 0.5 * (u ** 2 + v ** 2)
    # Vertical σ-integral (Σ_k Δσ_k · specific_energy_k).
    column = (specific_energy * d_sigma).sum(dim=-3)  # (..., x, y)

    # Mass-weight by Ps/g and integrate over the sphere (Gaussian quadrature).
    integrand = ps.squeeze(-3) * column / g  # (..., x, y)
    energy = grid.integrate(integrand)

    if surface_geopotential is not None:
        phi_s = surface_geopotential
        if phi_s.device != tvar.device:
            phi_s = phi_s.to(device=tvar.device)
        phi_s = phi_s.to(dtype=torch.float64)
        # Φ_s is a single 2-D field (x, y); collapse any singleton level/batch dims
        # so it broadcasts against the (..., x, y) mass-weighted surface pressure.
        while phi_s.dim() > 2:
            phi_s = phi_s.squeeze(0)
        energy = energy + grid.integrate(ps.squeeze(-3) * phi_s / g)

    return energy


# ═══════════════════════════════════════════════════════════════════════════
# Fixers
# ═══════════════════════════════════════════════════════════════════════════

class GlobalDryAirMassFixer(nn.Module):
    """Correct log_surface_pressure to conserve dry air mass.

    Two semantics are available:

    ``physical`` (default when a context is supplied)
        Conserve the mass integral ``∫ Ps/g dA``.  Adding ``delta`` to the
        (0,0) mode multiplies ``Ps`` by ``exp(delta)`` uniformly, so the exact
        correction is ``delta = log(M_before / M_after)``.

    ``proxy``
        Zero the (0,0) modal drift of ``log(Ps)`` -- the legacy behaviour, which
        conserves the *global mean of log(Ps)* rather than the mass.
    """

    def __init__(self, ctx=None, metric: str = "physical"):
        super().__init__()
        self.ctx = ctx
        self.metric = metric if ctx is not None else "proxy"
        # Grid constant read on every physical fix; captured at construction
        # so the traced forward stays free of data-dependent ops (ISSUE-016).
        self._unit00 = (unit00_nodal(ctx["coords"].horizontal)
                        if ctx is not None else None)
        self.last_correction = None
        self.last_metric = None

    def forward(
        self,
        state: primitive_equations.State,
        state_before: primitive_equations.State,
    ) -> primitive_equations.State:
        lsp = state.log_surface_pressure
        lsp_new = lsp.clone()

        if self.ctx is not None and self.metric == "physical":
            # Physical semantics: conserve the mass integral ∫ Ps/g dA.
            # Adding delta to the (0,0) mode multiplies Ps by exp(delta)
            # uniformly, so the exact correction is delta = log(M0/M1).
            m_before = compute_column_mass(state_before, self.ctx)
            m_after = compute_column_mass(state, self.ctx)
            d_node = torch.log(m_before / m_after)           # nodal log-Ps offset
            b = self._unit00
            self.last_correction = d_node.detach()
            lsp_new[..., 0, 0] = lsp_new[..., 0, 0] + (d_node / b).to(
                dtype=lsp_new.dtype)
            self.last_metric = "physical"
        else:
            # Legacy proxy semantics: zero the (0,0) mode drift of log(Ps).
            drift = lsp[..., 0, 0] - state_before.log_surface_pressure[..., 0, 0]
            self.last_correction = drift.detach()
            lsp_new[..., 0, 0] = lsp_new[..., 0, 0] - drift
            self.last_metric = "proxy"

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

    Two semantics are available:

    ``physical`` (default when a context is supplied)
        Conserve the mass-weighted column energy (:func:`compute_total_energy_physical`),
        the quantity the reference implementation controls, with
        ``dT = dE / (Cp · M)`` applied uniformly to every level's (0,0) mode.

    ``proxy``
        Conserve the raw spectral proxy ``Σ v² + Σ d² + Σ T'²`` via the
        quadratic solve below -- the legacy behaviour, kept so the numbers
        published with it stay reproducible.
    """

    def __init__(self, ctx=None, metric: str = "physical"):
        super().__init__()
        self.ctx = ctx
        self.metric = metric if ctx is not None else "proxy"
        # Grid constant read on every physical fix; captured at construction
        # so the traced forward stays free of data-dependent ops (ISSUE-016).
        self._unit00 = (unit00_nodal(ctx["coords"].horizontal)
                        if ctx is not None else None)
        self.last_correction = None
        self.last_metric = None

    def forward(
        self,
        state: primitive_equations.State,
        state_before: primitive_equations.State,
    ) -> primitive_equations.State:
        if self.ctx is not None and self.metric == "physical":
            # Physical semantics: conserve the mass-weighted column energy
            # E = ∫ Ps/g ∫(Cp·T + KE) dσ dA.  E is *linear* in a uniform
            # temperature offset, so the correction is exact in one shot:
            #   dT = (E_before - E_after) / (Cp · M),   M = ∫ Ps/g dA
            # which is the same form the reference implementation uses.
            e_before = compute_total_energy_physical(
                state_before.tree_map(lambda t: t.to(torch.float64)), **self.ctx)
            e_after = compute_total_energy_physical(
                state.tree_map(lambda t: t.to(torch.float64)), **self.ctx)
            mass = compute_column_mass(state, self.ctx)
            cp = float(self.ctx["physics_specs"].Cp)
            d_node = (e_before - e_after) / (cp * mass)      # nodal offset
            b = self._unit00
            self.last_correction = d_node.detach()           # report the nodal value
            self.last_metric = "physical"
            temp_fixed = state.temperature_variation.clone()
            temp_fixed[..., 0, 0] = temp_fixed[..., 0, 0] + (d_node / b).to(
                dtype=temp_fixed.dtype)
            return primitive_equations.State(
                vorticity=state.vorticity,
                divergence=state.divergence,
                temperature_variation=temp_fixed,
                log_surface_pressure=state.log_surface_pressure,
                tracers=state.tracers,
                sim_time=state.sim_time,
            )

        self.last_metric = "proxy"
        # Compute energy deficit (should be zero for perfect conservation)
        e_before = compute_total_energy(state_before)
        e_after = compute_total_energy(state)
        e_deficit = e_before - e_after  # positive means energy was lost

        # Correct via a SINGLE uniform (global-mean) temperature offset applied
        # to EVERY level's (0,0) spectral mode.  ``e_deficit`` is a GLOBAL scalar
        # (compute_total_energy sums over all levels+coeffs), so the correction
        # must also be a single scalar — not a per-level vector.  Let
        #   S = sum_k T[k,0,0]  (over the N (0,0) modes),  N = number of modes.
        # Adding the same delta to all N modes changes the proxy energy by
        #   dE = sum_k[(T_k+delta)^2 - T_k^2] = N*delta^2 + 2*S*delta.
        # Solving N*delta^2 + 2*S*delta - e_deficit = 0 for the root closest to 0:
        #   delta = (-S + sign(S)*sqrt(S^2 + N*e_deficit)) / N.
        # (The old code used a PER-LEVEL T_00 vector, so each level independently
        #  solved the FULL deficit -> ~N-fold over-correction -> oscillation.)
        T_00 = state.temperature_variation[..., 0, 0]
        S = T_00.sum()
        N = T_00.numel()
        discriminant = S ** 2 + N * e_deficit
        # Tensor-level guard — no data-dependent Python branch.
        # torch.where selects between the corrected (0,0) mode and the original,
        # preserving the full state unchanged when discriminant ≤ 0.
        needs_fix = discriminant > 0
        safe_discriminant = torch.clamp(discriminant, min=0.0)
        delta = (-S + torch.sign(S) * torch.sqrt(safe_discriminant)) / N

        temp_fixed = state.temperature_variation.clone()
        temp_fixed[..., 0, 0] = torch.where(
            needs_fix,
            temp_fixed[..., 0, 0] + delta,
            temp_fixed[..., 0, 0],
        )
        self.last_correction = delta.detach()

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

    def __init__(self, fix_mass: bool = True, fix_energy: bool = True,
                 dtype: torch.dtype = torch.float64, physical_ctx=None,
                 metric: str = "physical"):
        super().__init__()
        self.dtype = dtype  # informational; actual precision is set by Z2 policy via model
        # Without the physical context the fixers fall back to the legacy proxy
        # quantities and say so on ``last_metric`` -- never silently.
        self.physical_ctx = physical_ctx
        self.metric = metric
        self.mass_fixer = GlobalDryAirMassFixer(ctx=physical_ctx, metric=metric) if fix_mass else None
        self.energy_fixer = GlobalEnergyFixer(ctx=physical_ctx, metric=metric) if fix_energy else None

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

    @property
    def metrics_in_use(self) -> dict:
        """Which semantics each sub-fixer actually applied (audit surface)."""
        out = {}
        for name, fx in (("mass", self.mass_fixer), ("energy", self.energy_fixer)):
            if fx is not None:
                out[name] = getattr(fx, "last_metric", None)
        return out


def conservation_report(model, state, steps: int = 200, marks=None) -> dict:
    """Run ``model`` and report BOTH conservation quantities step by step.

    The proxy (what the fixer used to control) and the physical mass-weighted
    column energy (what the reference implementation controls) can disagree by
    orders of magnitude; reporting only one of them is how a fixer can look
    perfectly conserving while the model is not.

    Returns a dict with ``proxy`` and ``physical`` series plus the fixer's own
    recorded corrections.
    """
    import copy

    if marks is None:
        marks = [1, 10, 30, 60, 100, 200]
    marks = sorted(m for m in marks if m <= steps)

    ctx = None
    fixer = getattr(model, "conservation_fixer", None)
    if fixer is not None:
        ctx = getattr(fixer, "physical_ctx", None)
    if ctx is None:
        raise RuntimeError(
            "conservation_report needs a fixer built with physical_ctx "
            "(build_physical_ctx); refusing to report proxy-only"
        )

    s = copy.deepcopy(state)
    e0 = float(compute_total_energy(s.tree_map(lambda t: t.to(torch.float64))).cpu())
    p0 = float(compute_total_energy_physical(
        s.tree_map(lambda t: t.to(torch.float64)), **ctx).cpu())
    m0 = float(compute_column_mass(s, ctx).cpu())

    proxy, phys, mass, corr = {}, {}, {}, {}
    with torch.no_grad():
        for i in range(1, steps + 1):
            s = model.step(s)
            if i in marks:
                e = float(compute_total_energy(
                    s.tree_map(lambda t: t.to(torch.float64))).cpu())
                p = float(compute_total_energy_physical(
                    s.tree_map(lambda t: t.to(torch.float64)), **ctx).cpu())
                m = float(compute_column_mass(s, ctx).cpu())
                proxy[i] = abs(e - e0) / abs(e0)
                phys[i] = abs(p - p0) / abs(p0)
                mass[i] = abs(m - m0) / abs(m0)
                fx = getattr(model, "conservation_fixer", None)
                c = getattr(fx, "energy_fixer", None)
                if c is not None and c.last_correction is not None:
                    corr[i] = float(torch.as_tensor(c.last_correction).abs().max().cpu())
    return {"proxy": proxy, "physical": phys, "mass": mass,
            "fixer_dT": corr, "steps": steps, "marks": marks}


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

            # SINGLE uniform offset (see GlobalEnergyFixer): the deficit is a
            # global scalar, so delta must be a scalar shared by all (0,0) modes.
            #   delta = (-S + sign(S)*sqrt(S^2 + N*e_deficit)) / N.
            T_00 = state.temperature_variation[..., 0, 0].to(torch.float64)
            S = T_00.sum()
            N = T_00.numel()
            discriminant = S ** 2 + N * e_deficit
            if discriminant > 0:
                delta = (-S + torch.sign(S) * torch.sqrt(discriminant)) / N
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
