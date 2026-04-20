# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""Spectral sparse State representations.

Provides:
  - SparseConfig: all sparse mode parameters
  - MixedPrecisionSparse: low-order FP32 + high-order BF16 tensor pair
  - SpectralSparseState: sparse wrapper around State dataclass
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import torch

from tornado_gcm.precision.sparse.ops import triangular_mask


@dataclasses.dataclass
class SparseConfig:
    """Configuration for sparse spectral representations.

    Modes:
        structural — use triangular mask, store only valid SH positions
        adaptive — additionally zero out coefficients below threshold
        mixed — split low/high order with different dtypes
    """

    enabled: bool = False
    mode: str = "structural"  # 'structural' | 'adaptive' | 'mixed'
    adaptive_threshold: float = 1e-6
    high_order_cutoff: float = 0.5   # l > cutoff*L → high-order
    high_order_dtype: torch.dtype = torch.bfloat16
    low_order_dtype: torch.dtype = torch.float32


class MixedPrecisionSparse:
    """Mixed-precision split tensor: low-order FP32 + high-order BF16.

    Stores spectral coefficients in two blocks split at cutoff_l.
    Low-order (l < cutoff_l) remains FP32 for accuracy.
    High-order (l >= cutoff_l) is stored in BF16 for memory savings.
    """

    def __init__(
        self,
        low_order: torch.Tensor,
        high_order: torch.Tensor,
        cutoff_l: int,
    ):
        """
        Args:
            low_order: (..., M, cutoff_l), FP32.
            high_order: (..., M, L - cutoff_l), BF16.
            cutoff_l: split boundary (degree index).
        """
        self.low = low_order
        self.high = high_order
        self.cutoff_l = cutoff_l

    @classmethod
    def from_dense(
        cls,
        x: torch.Tensor,
        cutoff_l: int,
        high_dtype: torch.dtype = torch.bfloat16,
    ) -> "MixedPrecisionSparse":
        """Split a dense (..., M, L) tensor into low/high blocks."""
        low = x[..., :cutoff_l].to(torch.float32)
        high = x[..., cutoff_l:].to(high_dtype)
        return cls(low, high, cutoff_l)

    def to_dense(self) -> torch.Tensor:
        """Reconstruct the full (..., M, L) tensor in FP32."""
        return torch.cat([self.low, self.high.to(torch.float32)], dim=-1)

    @property
    def shape(self) -> tuple:
        """Full (non-split) shape."""
        M = self.low.shape[-2]
        L = self.low.shape[-1] + self.high.shape[-1]
        return (*self.low.shape[:-1], L)

    @property
    def device(self) -> torch.device:
        return self.low.device

    def memory_bytes(self) -> int:
        """Actual memory usage in bytes."""
        return (
            self.low.numel() * self.low.element_size()
            + self.high.numel() * self.high.element_size()
        )

    def dense_memory_bytes(self) -> int:
        """Memory if stored as dense FP32."""
        M = self.low.shape[-2]
        L = self.low.shape[-1] + self.high.shape[-1]
        batch = self.low[..., 0, 0].numel()
        return batch * M * L * 4

    @property
    def memory_savings(self) -> float:
        """Fraction saved vs dense FP32."""
        dense = self.dense_memory_bytes()
        if dense == 0:
            return 0.0
        return 1.0 - self.memory_bytes() / dense


class SpectralSparseState:
    """Sparse wrapper around a State-like object.

    Converts modal tensor fields in a State to sparse or mixed-precision
    representations. Supports round-trip: dense → sparse → dense.

    NOTE: autograd is NOT supported on sparse representations.
    Use to_dense() before entering training forward pass.
    """

    def __init__(
        self,
        fields: dict[str, MixedPrecisionSparse | torch.Tensor],
        config: SparseConfig,
        modal_shape: Optional[tuple[int, int]] = None,
    ):
        """
        Args:
            fields: dict mapping field name → MixedPrecisionSparse or
                     sparse COO tensor.
            config: sparse configuration.
            modal_shape: (M, L) shape for triangular masking.
        """
        self.fields = fields
        self.config = config
        self.modal_shape = modal_shape

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        config: SparseConfig,
    ) -> "SpectralSparseState":
        """Convert a dict of dense modal tensors to sparse representation.

        Args:
            state_dict: {'vorticity': Tensor(...,M,L), 'divergence': ..., ...}
            config: sparse configuration.

        Returns:
            SpectralSparseState with fields converted per config.mode.
        """
        fields: dict[str, MixedPrecisionSparse | torch.Tensor] = {}
        modal_shape = None

        for name, tensor in state_dict.items():
            if tensor.dim() < 2:
                # Scalar or 1D (e.g., sim_time) — keep dense
                fields[name] = tensor
                continue

            M, L = tensor.shape[-2], tensor.shape[-1]
            if modal_shape is None:
                modal_shape = (M, L)

            if config.mode == "structural":
                # Apply triangular mask, store as sparse COO
                mask = triangular_mask(M, L, device=tensor.device)
                masked = tensor * mask.unsqueeze(0) if tensor.dim() > 2 else tensor * mask
                # Store as regular tensor with zeros (COO not autograd-safe)
                fields[name] = masked

            elif config.mode == "adaptive":
                mask = triangular_mask(M, L, device=tensor.device)
                thresh_mask = tensor.abs() >= config.adaptive_threshold
                combined = mask & thresh_mask if tensor.dim() == 2 else (
                    mask.unsqueeze(0) & thresh_mask
                )
                fields[name] = tensor * combined

            elif config.mode == "mixed":
                cutoff_l = int(config.high_order_cutoff * L)
                fields[name] = MixedPrecisionSparse.from_dense(
                    tensor, cutoff_l, config.high_order_dtype
                )
            else:
                fields[name] = tensor

        return cls(fields, config, modal_shape)

    def to_dense_dict(self) -> dict[str, torch.Tensor]:
        """Reconstruct all fields as dense FP32 tensors."""
        result = {}
        for name, field in self.fields.items():
            if isinstance(field, MixedPrecisionSparse):
                result[name] = field.to_dense()
            else:
                result[name] = field.to(torch.float32) if field.is_floating_point() else field
        return result

    @property
    def memory_bytes(self) -> int:
        """Total memory across all fields."""
        total = 0
        for field in self.fields.values():
            if isinstance(field, MixedPrecisionSparse):
                total += field.memory_bytes()
            elif isinstance(field, torch.Tensor):
                total += field.numel() * field.element_size()
        return total

    @property
    def dense_memory_bytes(self) -> int:
        """Memory if all fields were dense FP32."""
        total = 0
        for field in self.fields.values():
            if isinstance(field, MixedPrecisionSparse):
                total += field.dense_memory_bytes()
            elif isinstance(field, torch.Tensor):
                total += field.numel() * 4  # FP32 baseline
        return total

    @property
    def memory_savings(self) -> float:
        """Fraction saved vs all-dense FP32."""
        dense = self.dense_memory_bytes
        if dense == 0:
            return 0.0
        return 1.0 - self.memory_bytes / dense
