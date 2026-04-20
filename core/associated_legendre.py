# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""Associated Legendre function evaluation and Gauss-Legendre quadrature.

Pure NumPy implementation (Z0 — FP64 numerical foundation).
Results are computed in float64 and intended to be stored as read-only buffers.
"""

from __future__ import annotations

import functools

import numpy as np
import scipy.special as sps


# ═══════════════════════════════════════════════════════════════════════════
# Internal: rhombus-indexed evaluation
# ═══════════════════════════════════════════════════════════════════════════

def _evaluate_rhombus(
    n_l: int,
    n_m: int,
    x: np.ndarray,
    truncation: str = "rhombus",
) -> np.ndarray:
    """Evaluate associated Legendre functions (rhombus indexing).

    Returns array p of shape (n_l, n_m, len(x)) such that
        p[k, m, i] = c_{k,m} P^m_{m+k}(x_i)
    with unit L²([-1, 1]) normalization.
    """
    y = np.sqrt(1 - x * x)
    p = np.zeros((n_l, n_m, len(x)), dtype=np.float64)
    p[0, 0] = 1 / np.sqrt(2)
    for m in range(1, n_m):
        p[0, m] = -np.sqrt(1 + 1 / (2 * m)) * y * p[0, m - 1]
    m_max = n_m
    for k in range(1, n_l):
        if truncation == "triangle":
            m_max = min(n_m, n_l - k)
        m = np.arange(m_max).reshape((-1, 1))
        m2 = np.square(m)
        mk2 = np.square(m + k)
        mkp2 = np.square(m + k - 1)
        a = np.sqrt((4 * mk2 - 1) / (mk2 - m2))
        b = np.sqrt((mkp2 - m2) / (4 * mkp2 - 1))
        p[k, :m_max] = a * (x * p[k - 1, :m_max] - b * p[k - 2, :m_max])
    return p


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(n_m: int, n_l: int, x: np.ndarray) -> np.ndarray:
    """Evaluate associated Legendre functions.

    Args:
        n_m: Number of azimuthal wavenumber modes. Must satisfy n_m <= n_l.
        n_l: Number of total wavenumber modes.
        x: Vector of nodes in [-1, 1].

    Returns:
        Array p of shape (n_m, len(x), n_l) such that
            p[m, i, l] = c_{l,m} P^m_l(x_i)
        with p[m, i, l] = 0 for l < m. Unit L²([-1, 1]) normalized.
    """
    if n_m > n_l:
        raise ValueError(
            f"Expected n_m <= n_l; got n_m = {n_m} and n_l = {n_l}."
        )
    r = np.transpose(
        _evaluate_rhombus(n_l=n_l, n_m=n_m, x=x, truncation="triangle"),
        (1, 2, 0),
    )
    p = np.zeros((n_m, len(x), n_l), dtype=np.float64)
    for m in range(n_m):
        p[m, :, m:n_l] = r[m, :, 0 : n_l - m]
    return p


def gauss_legendre_nodes(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre quadrature nodes and weights (float64)."""
    return sps.roots_legendre(n)


@functools.lru_cache(maxsize=128)
def equiangular_nodes(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Equally spaced latitude nodes and weights (float64)."""
    spacing = np.pi / n
    theta = np.linspace(-np.pi / 2 + spacing / 2, np.pi / 2 - spacing / 2, n)
    x = np.sin(theta)
    w = _compute_weights(x)
    return x, w


@functools.lru_cache(maxsize=128)
def equiangular_nodes_with_poles(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Equally spaced latitude nodes including poles and weights (float64)."""
    theta = np.linspace(-np.pi / 2, np.pi / 2, n)
    x = np.sin(theta)
    w = _compute_weights(x)
    return x, w


def _compute_weights(x: np.ndarray) -> np.ndarray:
    """Compute quadrature weights for nodes x via Legendre polynomial solve."""
    legendre = evaluate(n_m=1, n_l=x.shape[0], x=x)[0].T
    z = np.zeros_like(x)
    z[0] = 1
    w = np.linalg.solve(legendre, z)
    return w / w.sum() * 2
