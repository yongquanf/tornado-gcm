"""Sparse spectral representations for SDA framework.

Provides:
  - SparseConfig: configuration for sparse modes
  - SpectralSparseState: sparse wrapper around State
  - MixedPrecisionSparse: low-order FP32 + high-order BF16 tensor
  - SparseSpectralTransform: split-precision SHT
  - sparse_ops: mask generation and adaptive thresholding
"""

from pytorch_src.precision.sparse.spectral_sparse import (
    SparseConfig,
    SpectralSparseState,
    MixedPrecisionSparse,
)
from pytorch_src.precision.sparse.adaptive_sparse import (
    SparseSpectralTransform,
)
from pytorch_src.precision.sparse.ops import (
    triangular_mask,
    adaptive_threshold_mask,
    memory_savings_ratio,
)

__all__ = [
    "SparseConfig",
    "SpectralSparseState",
    "MixedPrecisionSparse",
    "SparseSpectralTransform",
    "triangular_mask",
    "adaptive_threshold_mask",
    "memory_savings_ratio",
]
