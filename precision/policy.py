"""PrecisionPolicy: configurable precision strategy for PZHA zones.

Defines the four precision zones (Z0-Z3) and their dtype assignments.
Driven by configuration (dataclass), not hardcoded.

Zones:
    Z0 — Numerical Foundation: FP64 (Legendre/Fourier bases, σ-coords, implicit matrices)
    Z1 — Dynamics Core: TF32/HIGHEST (SH transforms, primitive equation einsums)
    Z2 — Conservation Fixers: FP64 compute, FP32 storage (energy/mass fixers)
    Z3 — Neural Network: BF16 compute, FP32 master params (MLP, Transformer)
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Optional

import torch


class PrecisionZone(enum.Enum):
    """The four PZHA precision zones."""

    Z0_NUMERICAL_FOUNDATION = "Z0"
    Z1_DYNAMICS_CORE = "Z1"
    Z2_CONSERVATION_FIXER = "Z2"
    Z3_NEURAL_NETWORK = "Z3"


@dataclasses.dataclass
class PrecisionPolicy:
    """Configurable precision policy for PZHA zones.

    Attributes:
        z0_dtype: Z0 compute and storage dtype. Always float64.
        z1_compute_dtype: Z1 compute dtype (SH transforms, einsums).
            - 'float64' for HIGHEST precision
            - 'float32' for high-res safe default
            - 'tf32' uses float32 storage with TF32 matmul acceleration
        z1_storage_dtype: Z1 state storage dtype. Typically float32.
        z2_compute_dtype: Z2 fixer internal compute dtype. Typically float64.
        z2_storage_dtype: Z2 fixer output storage dtype. Typically float32.
        z3_compute_dtype: Z3 NN forward/backward dtype. Typically bfloat16.
        z3_param_dtype: Z3 NN parameter storage dtype. bfloat16 (inference) or float32 (master).
        z3_master_dtype: Z3 NN master parameter dtype (training). float32.
        data_load_dtype: Data pipeline dtype. float32 or bfloat16.
        enable_precision_audit: Whether to log precision loss at zone boundaries.
        audit_every_n_steps: Frequency of precision audits.
    """

    # Z0: Numerical foundation (always FP64, not normally overridden)
    z0_dtype: torch.dtype = torch.float64

    # Z1: Dynamics core
    z1_compute_dtype: torch.dtype = torch.float32
    z1_use_tf32: bool = True  # Enable TF32 matmul acceleration on GPU
    z1_storage_dtype: torch.dtype = torch.float32

    # Z2: Conservation fixers
    z2_compute_dtype: torch.dtype = torch.float64
    z2_storage_dtype: torch.dtype = torch.float32

    # Z3: Neural networks
    z3_compute_dtype: torch.dtype = torch.bfloat16
    z3_param_dtype: torch.dtype = torch.bfloat16
    z3_master_dtype: torch.dtype = torch.float32

    # Data pipeline
    data_load_dtype: torch.dtype = torch.float32

    # SHT adaptive precision threshold (Algorithm 8)
    sht_adaptive_threshold: int = 42  # L <= threshold → FP32/TF32; L > threshold → FP64

    # Monitoring
    enable_precision_audit: bool = True
    audit_every_n_steps: int = 100

    def compute_dtype(self, zone: PrecisionZone) -> torch.dtype:
        """Get the compute dtype for a given zone."""
        mapping = {
            PrecisionZone.Z0_NUMERICAL_FOUNDATION: self.z0_dtype,
            PrecisionZone.Z1_DYNAMICS_CORE: self.z1_compute_dtype,
            PrecisionZone.Z2_CONSERVATION_FIXER: self.z2_compute_dtype,
            PrecisionZone.Z3_NEURAL_NETWORK: self.z3_compute_dtype,
        }
        return mapping[zone]

    def storage_dtype(self, zone: PrecisionZone) -> torch.dtype:
        """Get the storage dtype for a given zone."""
        mapping = {
            PrecisionZone.Z0_NUMERICAL_FOUNDATION: self.z0_dtype,
            PrecisionZone.Z1_DYNAMICS_CORE: self.z1_storage_dtype,
            PrecisionZone.Z2_CONSERVATION_FIXER: self.z2_storage_dtype,
            PrecisionZone.Z3_NEURAL_NETWORK: self.z3_param_dtype,
        }
        return mapping[zone]

    def apply_tf32_setting(self) -> None:
        """Apply TF32 matmul setting globally for Z1 acceleration."""
        torch.backends.cuda.matmul.allow_tf32 = self.z1_use_tf32
        torch.backends.cudnn.allow_tf32 = self.z1_use_tf32

    @classmethod
    def for_resolution(cls, total_wavenumber: int) -> PrecisionPolicy:
        """Create a resolution-adaptive precision policy.

        Low-res (L<=42): more aggressive BF16 in Z1
        Mid-res (L<=127): standard TF32
        High-res (L>127): conservative float32 in Z1
        """
        if total_wavenumber <= 42:
            return cls(z1_compute_dtype=torch.float32, z1_use_tf32=True)
        elif total_wavenumber <= 127:
            return cls(z1_compute_dtype=torch.float32, z1_use_tf32=True)
        else:
            return cls(z1_compute_dtype=torch.float32, z1_use_tf32=False)

    @classmethod
    def for_training_phase(
        cls,
        phase: int,
        base: Optional["PrecisionPolicy"] = None,
    ) -> PrecisionPolicy:
        """Create policy for variable-precision training phases.

        Phase 1: Aggressive low-precision (NN=BF16, dycore=BF16/TF32, fixer=FP32)
        Phase 2: Standard PZHA (NN=BF16, dycore=TF32, fixer=FP64)
        Phase 3: High-precision fine-tune (NN=FP32, dycore=TF32, fixer=FP64)
        """
        if base is None:
            base = cls()

        if phase == 1:
            return dataclasses.replace(
                base,
                z3_compute_dtype=torch.bfloat16,
                z1_use_tf32=True,
                z2_compute_dtype=torch.float32,
            )
        elif phase == 2:
            return dataclasses.replace(
                base,
                z3_compute_dtype=torch.bfloat16,
                z1_use_tf32=True,
                z2_compute_dtype=torch.float64,
            )
        elif phase == 3:
            return dataclasses.replace(
                base,
                z3_compute_dtype=torch.float32,
                z3_param_dtype=torch.float32,
                z1_use_tf32=True,
                z2_compute_dtype=torch.float64,
            )
        else:
            raise ValueError(f"Unknown training phase: {phase}")


# ---------------------------------------------------------------------------
# PolicyEngine: SDA-aware policy management
# ---------------------------------------------------------------------------


class PolicyEngine:
    """Policy compilation and runtime management for SDA.

    Wraps a PrecisionPolicy with:
      - compile() → fast lookup table
      - hot_swap() → runtime policy replacement
      - get_module_zone_map() → auto-assign zones to model submodules

    The engine maintains backward compatibility: any code using a plain
    PrecisionPolicy is unaffected.
    """

    def __init__(self, policy: PrecisionPolicy | None = None) -> None:
        self._policy = policy or PrecisionPolicy()

    @property
    def policy(self) -> PrecisionPolicy:
        return self._policy

    def hot_swap(self, new_policy: PrecisionPolicy) -> None:
        """Replace the active policy at a step boundary."""
        self._policy = new_policy
        new_policy.apply_tf32_setting()

    def get_zone_dtype_map(self) -> dict[PrecisionZone, tuple[torch.dtype, torch.dtype]]:
        """Return {zone: (compute_dtype, storage_dtype)} for all zones."""
        return {
            z: (self._policy.compute_dtype(z), self._policy.storage_dtype(z))
            for z in PrecisionZone
        }

    def get_module_zone_map(
        self, model: torch.nn.Module
    ) -> dict[str, PrecisionZone]:
        """Auto-infer which zone each submodule belongs to.

        Heuristic:
          - Modules whose name contains 'nn', 'mlp', 'transformer', 'encoder',
            'decoder', 'learned' → Z3
          - Modules whose name contains 'fixer', 'conservation' → Z2
          - Modules whose name contains 'sht', 'spectral', 'legendre', 'basis' → Z0
          - Everything else → Z1
        """
        zone_map: dict[str, PrecisionZone] = {}
        z3_keywords = {"nn", "mlp", "transformer", "encoder", "decoder", "learned", "embed"}
        z2_keywords = {"fixer", "conservation", "energy", "mass"}
        z0_keywords = {"sht", "spectral", "legendre", "basis", "fourier", "sigma"}

        for name, _ in model.named_modules():
            lower = name.lower()
            if any(kw in lower for kw in z3_keywords):
                zone_map[name] = PrecisionZone.Z3_NEURAL_NETWORK
            elif any(kw in lower for kw in z2_keywords):
                zone_map[name] = PrecisionZone.Z2_CONSERVATION_FIXER
            elif any(kw in lower for kw in z0_keywords):
                zone_map[name] = PrecisionZone.Z0_NUMERICAL_FOUNDATION
            else:
                zone_map[name] = PrecisionZone.Z1_DYNAMICS_CORE
        return zone_map


# ---------------------------------------------------------------------------
# Convenience: default policy singleton
# ---------------------------------------------------------------------------
DEFAULT_POLICY = PrecisionPolicy()
