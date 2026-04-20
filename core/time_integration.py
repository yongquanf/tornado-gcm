# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""Implicit–explicit time integration routines — PyTorch implementation.

This module provides IMEX ODE solvers (Crank-Nicolson + RK, IMEX RK, etc.)
for spectral atmospheric dynamics. The `tree_math.struct` pattern from JAX
is replaced by plain pytree operations using `torch` tree_map equivalents.

Zone: Z1 (dynamics compute) — all time-stepping arithmetic in Z1 precision.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Callable, Optional, Sequence, TypeVar, Union

import torch

from tornado_gcm.core import filtering
from tornado_gcm.core import spherical_harmonic


# ═══════════════════════════════════════════════════════════════════════════
# Type aliases
# ═══════════════════════════════════════════════════════════════════════════

PyTreeState = Any
PyTreeTermsFn = Callable[[PyTreeState], PyTreeState]
PyTreeInverseFn = Callable[[PyTreeState, float], PyTreeState]
TimeStepFn = Callable[[PyTreeState], PyTreeState]
PyTreeStepFilterFn = Callable[[PyTreeState, PyTreeState], PyTreeState]
PostProcessFn = Callable[[PyTreeState], Any]


# ═══════════════════════════════════════════════════════════════════════════
# Pytree utilities (replacing JAX tree_math)
# ═══════════════════════════════════════════════════════════════════════════

def _tree_map(fn, *trees):
    """Apply fn element-wise across matching pytree leaves."""
    first = trees[0]
    if isinstance(first, torch.Tensor):
        return fn(*trees)
    elif isinstance(first, dict):
        return {k: _tree_map(fn, *[t[k] for t in trees]) for k in first}
    elif isinstance(first, (list, tuple)):
        result = [_tree_map(fn, *[t[i] for t in trees]) for i in range(len(first))]
        return type(first)(result)
    elif hasattr(first, '__dataclass_fields__'):
        fields = {f.name: _tree_map(fn, *[getattr(t, f.name) for t in trees])
                  for f in dataclasses.fields(first)
                  if isinstance(getattr(first, f.name), (torch.Tensor, dict, list, tuple))
                  or hasattr(getattr(first, f.name), '__dataclass_fields__')}
        return dataclasses.replace(first, **fields)
    else:
        return first  # scalars, None, etc.


def _tree_add(a, b):
    return _tree_map(torch.add, a, b)


def _tree_scale(tree, c: float):
    return _tree_map(lambda x: x * c, tree)


def _tree_neg(tree):
    return _tree_map(torch.neg, tree)


# ═══════════════════════════════════════════════════════════════════════════
# ODE equation interfaces
# ═══════════════════════════════════════════════════════════════════════════

class ExplicitODE:
    """ODE with only explicit terms."""

    def explicit_terms(self, state: PyTreeState) -> PyTreeState:
        raise NotImplementedError

    @classmethod
    def from_functions(cls, explicit_terms: PyTreeTermsFn) -> ExplicitODE:
        ode = cls()
        ode.explicit_terms = explicit_terms
        return ode


class ImplicitExplicitODE:
    """ODE with implicit and explicit terms: ∂x/∂t = F(x) + G(x)."""

    def explicit_terms(self, state: PyTreeState) -> PyTreeState:
        raise NotImplementedError

    def implicit_terms(self, state: PyTreeState) -> PyTreeState:
        raise NotImplementedError

    def implicit_inverse(self, state: PyTreeState, step_size: float) -> PyTreeState:
        """Apply (I - step_size * G)⁻¹ to state."""
        raise NotImplementedError

    @classmethod
    def from_functions(
        cls,
        explicit_terms: PyTreeTermsFn,
        implicit_terms: PyTreeTermsFn,
        implicit_inverse: PyTreeInverseFn,
    ) -> ImplicitExplicitODE:
        ode = cls()
        ode.explicit_terms = explicit_terms
        ode.implicit_terms = implicit_terms
        ode.implicit_inverse = implicit_inverse
        return ode


@dataclasses.dataclass
class TimeReversedImExODE(ImplicitExplicitODE):
    """ImplicitExplicitODE reversed in time."""
    forward_eq: ImplicitExplicitODE

    def explicit_terms(self, state):
        return _tree_neg(self.forward_eq.explicit_terms(state))

    def implicit_terms(self, state):
        return _tree_neg(self.forward_eq.implicit_terms(state))

    def implicit_inverse(self, state, step_size):
        return self.forward_eq.implicit_inverse(state, -step_size)


def compose_equations(
    equations: Sequence[Union[ImplicitExplicitODE, ExplicitODE]],
) -> ImplicitExplicitODE:
    """Combine equations with at most one ImplicitExplicitODE."""
    imex = [e for e in equations if isinstance(e, ImplicitExplicitODE)]
    if len(imex) != 1:
        raise ValueError(
            f"Expected exactly 1 ImplicitExplicitODE, got {len(imex)}"
        )
    (imex_eq,) = imex

    def explicit_fn(x):
        terms = [eq.explicit_terms(x) for eq in equations]
        result = terms[0]
        for t in terms[1:]:
            result = _tree_add(result, t)
        return result

    return ImplicitExplicitODE.from_functions(
        explicit_fn, imex_eq.implicit_terms, imex_eq.implicit_inverse
    )


# ═══════════════════════════════════════════════════════════════════════════
# IMEX time-stepping schemes
# ═══════════════════════════════════════════════════════════════════════════

def backward_forward_euler(
    equation: ImplicitExplicitODE,
    time_step: float,
) -> TimeStepFn:
    """First-order: forward Euler (explicit) + backward Euler (implicit)."""
    dt = time_step

    def step_fn(u0):
        g = _tree_add(u0, _tree_scale(equation.explicit_terms(u0), dt))
        return equation.implicit_inverse(g, dt)
    return step_fn


def crank_nicolson_rk2(
    equation: ImplicitExplicitODE,
    time_step: float,
) -> TimeStepFn:
    """Second-order: Crank-Nicolson (implicit) + Heun RK2 (explicit)."""
    dt = time_step

    def step_fn(u0):
        g = _tree_add(u0, _tree_scale(equation.implicit_terms(u0), 0.5 * dt))
        h1 = equation.explicit_terms(u0)
        u1 = equation.implicit_inverse(_tree_add(g, _tree_scale(h1, dt)), 0.5 * dt)
        h2_raw = equation.explicit_terms(u1)
        h2 = _tree_map(lambda a, b: 0.5 * (a + b), h2_raw, h1)
        return equation.implicit_inverse(_tree_add(g, _tree_scale(h2, dt)), 0.5 * dt)
    return step_fn


def low_storage_runge_kutta_crank_nicolson(
    alphas: Sequence[float],
    betas: Sequence[float],
    gammas: Sequence[float],
    equation: ImplicitExplicitODE,
    time_step: float,
) -> TimeStepFn:
    """Low-storage RK + Crank-Nicolson IMEX scheme."""
    dt = time_step
    if len(alphas) - 1 != len(betas) or len(betas) != len(gammas):
        raise ValueError("RK coefficient lengths don't match")

    def step_fn(u):
        h = None
        for k in range(len(betas)):
            f_u = equation.explicit_terms(u)
            if h is None:
                h = f_u
            else:
                h = _tree_add(f_u, _tree_scale(h, betas[k]))
            mu = 0.5 * dt * (alphas[k + 1] - alphas[k])
            rhs = _tree_add(
                _tree_add(u, _tree_scale(h, gammas[k] * dt)),
                _tree_scale(equation.implicit_terms(u), mu),
            )
            u = equation.implicit_inverse(rhs, mu)
        return u
    return step_fn


def crank_nicolson_rk3(
    equation: ImplicitExplicitODE,
    time_step: float,
) -> TimeStepFn:
    """CN + RK3 (Williamson)."""
    return low_storage_runge_kutta_crank_nicolson(
        alphas=[0, 1/3, 3/4, 1],
        betas=[0, -5/9, -153/128],
        gammas=[1/3, 15/16, 8/15],
        equation=equation,
        time_step=time_step,
    )


def crank_nicolson_rk4(
    equation: ImplicitExplicitODE,
    time_step: float,
) -> TimeStepFn:
    """CN + RK4 (Carpenter-Kennedy)."""
    return low_storage_runge_kutta_crank_nicolson(
        alphas=[0, 0.1496590219993, 0.3704009573644, 0.6222557631345,
                0.9582821306748, 1],
        betas=[0, -0.4178904745, -1.192151694643, -1.697784692471,
               -1.514183444257],
        gammas=[0.1496590219993, 0.3792103129999, 0.8229550293869,
                0.6994504559488, 0.1530572479681],
        equation=equation,
        time_step=time_step,
    )


# ── IMEX Butcher tableau ─────────────────────────────────────────────────

@dataclasses.dataclass
class ImExButcherTableau:
    a_ex: Sequence[Sequence[float]]
    a_im: Sequence[Sequence[float]]
    b_ex: Sequence[float]
    b_im: Sequence[float]


def imex_runge_kutta(
    tableau: ImExButcherTableau,
    equation: ImplicitExplicitODE,
    time_step: float,
    step_hook: Optional[Callable[[int, Any], None]] = None,
) -> TimeStepFn:
    """General IMEX Runge-Kutta time stepping.

    Args:
        tableau: IMEX Butcher tableau coefficients.
        equation: ImplicitExplicitODE to integrate.
        time_step: Integration time step dt.
        step_hook: Optional callable ``(stage_index, y_i) -> None`` invoked
            after each intermediate RK stage with the stage index and the
            intermediate state.  Useful for diagnostic monitoring (e.g.
            Nanoscope) without modifying the main computation path.
    """
    dt = time_step
    a_ex, a_im = tableau.a_ex, tableau.a_im
    b_ex, b_im = tableau.b_ex, tableau.b_im
    n = len(b_ex)

    def step_fn(y0):
        f = [None] * n
        g = [None] * n
        f[0] = equation.explicit_terms(y0)
        g[0] = equation.implicit_terms(y0)

        for i in range(1, n):
            ex = _tree_scale(y0, 0)  # zero
            for j in range(i):
                if a_ex[i-1][j]:
                    ex = _tree_add(ex, _tree_scale(f[j], dt * a_ex[i-1][j]))
            im = _tree_scale(y0, 0)
            for j in range(i):
                if a_im[i-1][j]:
                    im = _tree_add(im, _tree_scale(g[j], dt * a_im[i-1][j]))
            y_star = _tree_add(_tree_add(y0, ex), im)
            y_i = equation.implicit_inverse(y_star, dt * a_im[i-1][i])
            if step_hook is not None:
                step_hook(i, y_i)
            needs_f_i = b_ex[i] or any(a_ex[row][i] for row in range(i, n - 1))
            needs_g_i = b_im[i] or any(a_im[row][i] for row in range(i, n - 1))
            if needs_f_i:
                f[i] = equation.explicit_terms(y_i)
            if needs_g_i:
                g[i] = equation.implicit_terms(y_i)

        result = y0
        for j in range(n):
            if b_ex[j] and f[j] is not None:
                result = _tree_add(result, _tree_scale(f[j], dt * b_ex[j]))
            if b_im[j] and g[j] is not None:
                result = _tree_add(result, _tree_scale(g[j], dt * b_im[j]))
        return result
    return step_fn


def imex_rk_sil3(
    equation: ImplicitExplicitODE,
    time_step: float,
    step_hook: Optional[Callable[[int, Any], None]] = None,
) -> TimeStepFn:
    """SIL3 IMEX RK (Whitaker & Kar 2013)."""
    return imex_runge_kutta(
        tableau=ImExButcherTableau(
            a_ex=[[1/3], [1/6, 1/2], [1/2, -1/2, 1]],
            a_im=[[1/6, 1/6], [1/3, 0, 1/3], [3/8, 0, 3/8, 1/4]],
            b_ex=[1/2, -1/2, 1, 0],
            b_im=[3/8, 0, 3/8, 1/4],
        ),
        equation=equation,
        time_step=time_step,
        step_hook=step_hook,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Filters
# ═══════════════════════════════════════════════════════════════════════════

def runge_kutta_step_filter(state_filter: PyTreeTermsFn) -> PyTreeStepFilterFn:
    def _filter(u, u_next):
        return state_filter(u_next)
    return _filter


def exponential_step_filter(
    grid: spherical_harmonic.Grid,
    dt: float,
    tau: float = 0.010938,
    order: int = 18,
    cutoff: float = 0,
) -> PyTreeStepFilterFn:
    """Exponential step filter."""
    filter_fn = filtering.exponential_filter(grid, dt / tau, order, cutoff)
    return runge_kutta_step_filter(filter_fn)


def horizontal_diffusion_step_filter(
    grid: spherical_harmonic.Grid,
    dt: float,
    tau: float,
    order: int = 1,
) -> PyTreeStepFilterFn:
    """Horizontal diffusion step filter."""
    eigenvalues = grid.laplacian_eigenvalues
    scale = dt / (tau * abs(eigenvalues[-1]) ** order)
    filter_fn = filtering.horizontal_diffusion_filter(grid, scale, order)
    return runge_kutta_step_filter(filter_fn)


# ═══════════════════════════════════════════════════════════════════════════
# Trajectory utilities
# ═══════════════════════════════════════════════════════════════════════════

def step_with_filters(
    step_fn: TimeStepFn,
    filters: Sequence[PyTreeStepFilterFn],
) -> TimeStepFn:
    """Wrap step_fn with sequential filter application."""
    def _step_fn(u):
        u_next = step_fn(u)
        for filt in filters:
            u_next = filt(u, u_next)
        return u_next
    return _step_fn


def repeated(fn: TimeStepFn, steps: int) -> TimeStepFn:
    """Apply fn repeatedly for `steps` iterations."""
    if steps == 1:
        return fn

    def f_repeated(x):
        for _ in range(steps):
            x = fn(x)
        return x
    return f_repeated


def trajectory_from_step(
    step_fn: TimeStepFn,
    outer_steps: int,
    inner_steps: int,
    *,
    start_with_input: bool = False,
    post_process_fn: PostProcessFn = lambda x: x,
) -> Callable[[PyTreeState], tuple[PyTreeState, list]]:
    """Accumulate trajectory from repeated step_fn applications.

    Returns (final_state, trajectory_list_of_length_outer_steps).
    """
    if inner_steps != 1:
        step_fn = repeated(step_fn, inner_steps)

    def multistep(x):
        trajectory = []
        for _ in range(outer_steps):
            if start_with_input:
                trajectory.append(post_process_fn(x))
            x = step_fn(x)
            if not start_with_input:
                trajectory.append(post_process_fn(x))
        return x, trajectory
    return multistep
