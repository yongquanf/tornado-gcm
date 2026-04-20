# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""Sparse-aware spectral transform with split-precision Legendre.

SparseSpectralTransform wraps a RealSphericalHarmonics and performs
inverse/forward transforms using split low-order (FP32) and high-order
(BF16) Legendre basis blocks for memory-efficient inference.
"""

from __future__ import annotations

from typing import Optional

import torch

from tornado_gcm.precision.sparse.spectral_sparse import (
    MixedPrecisionSparse,
    SparseConfig,
)


class SparseSpectralTransform:
    """Split-precision spherical harmonic transform.

    Pre-splits the Legendre basis matrix P[M, J, L] into:
        P_low  = P[:, :, :cutoff]       — used with FP32 coefficients
        P_high = P[:, :, cutoff:]        — used with BF16 coefficients

    Inverse transform:
        y_low  = einsum("mjl,...ml->...mj", P_low,  x_low)   — FP64 acc
        y_high = einsum("mjl,...ml->...mj", P_high, x_high)  — FP32 acc (sufficient for high-l)
        y = y_low + y_high

    Args:
        legendre_basis: (M, J, L) FP64 Legendre polynomial matrix.
        config: sparse configuration (for cutoff and dtypes).
    """

    def __init__(
        self,
        legendre_basis: torch.Tensor,
        config: SparseConfig,
        fourier_basis: Optional[torch.Tensor] = None,
    ):
        L = legendre_basis.shape[2]
        self.cutoff_l = int(config.high_order_cutoff * L)
        self.config = config

        self.P_low = legendre_basis[:, :, :self.cutoff_l].clone()  # FP64
        self.P_high = legendre_basis[:, :, self.cutoff_l:].clone()  # FP64
        self.fourier_basis = fourier_basis  # optional (M, J) fourier matrix

    def inverse_legendre(
        self,
        sparse_coeff: MixedPrecisionSparse,
    ) -> torch.Tensor:
        """Split-precision inverse Legendre step.

        Computes: out[...,m,j] = sum_l P[m,j,l] * x[...,m,l]

        Low-order block uses FP64 accumulation for precision.
        High-order block uses FP32 (sufficient for small high-l coefficients).

        Returns:
            (..., M, J) nodal values in FP32.
        """
        # Low-order: FP64 accumulation
        P_low_64 = self.P_low.to(torch.float64)
        x_low_64 = sparse_coeff.low.to(torch.float64)
        y_low = torch.einsum("mjl,...ml->...mj", P_low_64, x_low_64)

        # High-order: FP32 (upcast BF16 high-order to FP32)
        P_high_32 = self.P_high.to(torch.float32)
        x_high_32 = sparse_coeff.high.to(torch.float32)
        y_high = torch.einsum("mjl,...ml->...mj", P_high_32, x_high_32)

        return (y_low + y_high).to(torch.float32)

    def forward_legendre(
        self,
        x: torch.Tensor,
    ) -> MixedPrecisionSparse:
        """Split-precision forward Legendre step.

        Computes: out[...,m,l] = sum_j P[m,j,l] * x[...,m,j]
        and splits result into low/high order blocks.

        Args:
            x: (..., M, J) nodal values.

        Returns:
            MixedPrecisionSparse with low (FP32) and high (BF16) blocks.
        """
        # Low-order: FP64 accumulation
        P_low_64 = self.P_low.to(torch.float64)
        x_64 = x.to(torch.float64)
        coeff_low = torch.einsum("mjl,...mj->...ml", P_low_64, x_64).to(torch.float32)

        # High-order: FP32 sufficient
        P_high_32 = self.P_high.to(torch.float32)
        x_32 = x.to(torch.float32)
        coeff_high = torch.einsum("mjl,...mj->...ml", P_high_32, x_32).to(
            self.config.high_order_dtype
        )

        return MixedPrecisionSparse(coeff_low, coeff_high, self.cutoff_l)

    def inverse_transform(
        self,
        sparse_coeff: MixedPrecisionSparse,
    ) -> torch.Tensor:
        """Full inverse SHT: Legendre then Fourier.

        Args:
            sparse_coeff: (..., M, L) in mixed precision.

        Returns:
            (..., n_lon, n_lat) nodal field in FP32.
        """
        # Step 1: Legendre (modal → intermediate)
        intermediate = self.inverse_legendre(sparse_coeff)  # (..., M, J)

        # Step 2: Fourier (if basis available)
        if self.fourier_basis is not None:
            # "im,...mj->...ij"
            F = self.fourier_basis.to(intermediate.dtype)
            return torch.einsum("im,...mj->...ij", F, intermediate)

        return intermediate

    def forward_transform(
        self,
        x: torch.Tensor,
    ) -> MixedPrecisionSparse:
        """Full forward SHT: Fourier then Legendre.

        Args:
            x: (..., n_lon, n_lat) nodal field.

        Returns:
            MixedPrecisionSparse coefficients.
        """
        # Step 1: Fourier (if basis available)
        if self.fourier_basis is not None:
            # "im,...ij->...mj"
            F = self.fourier_basis.to(x.dtype)
            intermediate = torch.einsum("im,...ij->...mj", F, x)
        else:
            intermediate = x

        # Step 2: Legendre
        return self.forward_legendre(intermediate)

    @property
    def memory_savings_estimate(self) -> dict[str, float]:
        """Estimate memory savings for a single field transform."""
        M, J, L_low = self.P_low.shape
        L_high = self.P_high.shape[2]
        L = L_low + L_high

        dense_fp32_per_field = M * L * 4
        mixed_per_field = M * L_low * 4 + M * L_high * 2  # FP32 + BF16

        return {
            "dense_fp32_bytes_per_field": dense_fp32_per_field,
            "mixed_bytes_per_field": mixed_per_field,
            "savings_fraction": 1.0 - mixed_per_field / dense_fp32_per_field,
            "cutoff_l": self.cutoff_l,
            "L_total": L,
        }
