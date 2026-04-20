# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""Accelerator backend package for SDA mixed-precision framework.

Provides:
  - AcceleratorRegistry: name → implementation mapping with priority
  - FusedAdvanceKernel: torch.compile wrapped model.step() (O9-1)
  - compile_backend: torch._dynamo configuration + pytree registration
  - triton_sht: Triton / PyTorch SHT Legendre kernels (O9-3)
  - triton_fixers: Triton / PyTorch conservation reduction kernels
"""

from tornado_gcm.precision.accelerator.registry import (
    AcceleratorRegistry,
    get_global_registry,
)
from tornado_gcm.precision.accelerator.fused_zone_cast import FusedAdvanceKernel
from tornado_gcm.precision.accelerator.compile_backend import (
    configure_dynamo,
    configure_matmul_precision,
    register_state_pytree,
    benchmark_compile,
)
from tornado_gcm.precision.accelerator.triton_sht import (
    sht_legendre_inverse,
    sht_legendre_forward,
    sht_legendre_inverse_torch,
    sht_legendre_forward_torch,
)
from tornado_gcm.precision.accelerator.triton_fixers import (
    sphere_integrate,
    spectral_norm,
    sphere_integrate_torch,
    spectral_norm_torch,
    KahanAccumulator,
)

__all__ = [
    "AcceleratorRegistry",
    "get_global_registry",
    "FusedAdvanceKernel",
    "configure_dynamo",
    "configure_matmul_precision",
    "register_state_pytree",
    "benchmark_compile",
    # Triton SHT
    "sht_legendre_inverse",
    "sht_legendre_forward",
    "sht_legendre_inverse_torch",
    "sht_legendre_forward_torch",
    # Triton fixers
    "sphere_integrate",
    "spectral_norm",
    "sphere_integrate_torch",
    "spectral_norm_torch",
    "KahanAccumulator",
]
