# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""Sparse spectral operations: masks, thresholding, memory estimation.

Spherical harmonic coefficient matrices have a natural triangular
sparsity pattern: for order m and degree l, only |m| <= l positions
are nonzero. This module provides utilities to exploit that structure.
"""

from __future__ import annotations

import torch


def triangular_mask(M: int, L: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Create the triangular validity mask for SH coefficients.

    For modal_shape (M, L) where M = 2*n_wavenumbers - 1 and
    L = n_wavenumbers + 1, position (m_idx, l) is valid when
    |order(m_idx)| <= l.

    Args:
        M: number of zonal wavenumber slots (2*n_wn - 1).
        L: number of total wavenumber slots (n_wn + 1).
        device: output device.

    Returns:
        Boolean mask, shape (M, L), True where coefficient is valid.
    """
    n_wn = (M + 1) // 2  # n_wavenumbers
    mask = torch.zeros(M, L, dtype=torch.bool, device=device)
    for m_idx in range(M):
        # Map index to signed order: 0..n_wn-1 → 0..n_wn-1 ; n_wn..M-1 → -(M-n_wn)...-1
        if m_idx < n_wn:
            order = m_idx
        else:
            order = m_idx - M
        abs_order = abs(order)
        for l in range(L):
            if abs_order <= l:
                mask[m_idx, l] = True
    return mask


def adaptive_threshold_mask(
    x: torch.Tensor,
    threshold: float = 1e-6,
) -> torch.Tensor:
    """Create a mask for coefficients above the absolute threshold.

    Args:
        x: modal tensor, shape (..., M, L).
        threshold: absolute value threshold.

    Returns:
        Boolean mask, shape (..., M, L), True where |x| >= threshold.
    """
    return x.abs() >= threshold


def count_structural_zeros(M: int, L: int) -> tuple[int, int]:
    """Count structural zeros in the triangular SH layout.

    Returns:
        (n_valid, n_total) tuple.
    """
    mask = triangular_mask(M, L)
    n_valid = mask.sum().item()
    n_total = M * L
    return int(n_valid), int(n_total)


def memory_savings_ratio(
    M: int,
    L: int,
    mode: str = "structural",
    high_order_cutoff: float = 0.5,
) -> float:
    """Estimate memory savings for the given sparse mode.

    Args:
        M: zonal wavenumber count.
        L: total wavenumber count.
        mode: 'structural', 'mixed', or 'structural+mixed'.
        high_order_cutoff: fraction of L above which = high-order.

    Returns:
        Fraction of original FP32 memory that is saved (0.0 = no savings, 1.0 = 100%).
    """
    n_valid, n_total = count_structural_zeros(M, L)
    cutoff_l = int(high_order_cutoff * L)
    fp32_bytes = 4

    if mode == "structural":
        # Only store valid entries (still FP32)
        return 1.0 - n_valid / n_total

    elif mode == "mixed":
        # Full dense but high-order in BF16, low-order in FP32
        low_count = M * cutoff_l
        high_count = M * (L - cutoff_l)
        mixed_bytes = low_count * 4 + high_count * 2
        dense_bytes = n_total * fp32_bytes
        return 1.0 - mixed_bytes / dense_bytes

    elif mode == "structural+mixed":
        # Sparse + mixed precision
        mask = triangular_mask(M, L)
        low_mask = mask.clone()
        low_mask[:, cutoff_l:] = False
        high_mask = mask.clone()
        high_mask[:, :cutoff_l] = False
        n_low = low_mask.sum().item()
        n_high = high_mask.sum().item()
        sparse_mixed_bytes = n_low * 4 + n_high * 2
        dense_bytes = n_total * fp32_bytes
        return 1.0 - sparse_mixed_bytes / dense_bytes

    return 0.0
