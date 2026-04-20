"""SDA-enhanced inference runner.

Extends MPInference with:
  - SDA control-plane integration (auto precision configuration)
  - Adaptive-precision inference (run_adaptive)
  - Policy hot-swap on quality degradation
  - Sparse spectral inference (run_sparse)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import torch

from pytorch_src.inference.runner import MPInference
from pytorch_src.precision.policy import PrecisionPolicy, PrecisionZone
from pytorch_src.precision.sda import SDAConfig, SDAController

logger = logging.getLogger(__name__)


# High-precision fallback policy (Z1 → FP32 no TF32)
_HIGH_PRECISION_POLICY = PrecisionPolicy(
    z1_compute_dtype=torch.float32,
    z1_use_tf32=False,
    z2_compute_dtype=torch.float64,
    z3_compute_dtype=torch.bfloat16,
)


class SDAInference(MPInference):
    """SDA-enhanced inference engine.

    Adds:
      - Aggressive BF16 parameter storage (no master weights needed)
      - SDA profiler for runtime quality monitoring
      - ``run_adaptive()``: auto-escalate precision when quality degrades

    Args:
        model: trained NeuralGCMModel (or InferenceModel).
        sda_config: SDAConfig (policy is adapted for inference).
    """

    def __init__(
        self,
        model: Any,
        sda_config: SDAConfig | None = None,
    ) -> None:
        self.sda_config = sda_config or SDAConfig()

        # Build inference-optimised policy: BF16 params, no master weights
        base = self.sda_config.policy
        infer_policy = PrecisionPolicy(
            z0_dtype=base.z0_dtype,
            z1_compute_dtype=base.z1_compute_dtype,
            z1_use_tf32=base.z1_use_tf32,
            z1_storage_dtype=base.z1_storage_dtype,
            z2_compute_dtype=base.z2_compute_dtype,
            z2_storage_dtype=base.z2_storage_dtype,
            z3_compute_dtype=torch.bfloat16,
            z3_param_dtype=torch.bfloat16,
            z3_master_dtype=torch.bfloat16,  # no master needed
            data_load_dtype=torch.bfloat16,
            sht_adaptive_threshold=base.sht_adaptive_threshold,
            enable_precision_audit=base.enable_precision_audit,
        )

        self.controller = SDAController(
            SDAConfig(
                policy=infer_policy,
                profiler=self.sda_config.profiler,
                scheduler=self.sda_config.scheduler,
            )
        )
        super().__init__(model, infer_policy, monitor=self.controller.profiler)
        self.controller.apply()

    # ── Standard run (override to add SDA) ──

    @torch.no_grad()
    def run(
        self,
        initial_state: Any,
        outer_steps: int,
        inner_steps: int | None = None,
        post_process_fn: Callable | None = None,
        forcings: dict[str, torch.Tensor] | None = None,
    ) -> list[Any]:
        """Run inference with SDA monitoring."""
        trajectory = super().run(
            initial_state, outer_steps, inner_steps,
            post_process_fn=post_process_fn,
            forcings=forcings,
        )
        return trajectory

    # ── Adaptive-precision run ──

    @torch.no_grad()
    def run_adaptive(
        self,
        initial_state: Any,
        outer_steps: int,
        inner_steps: int = 1,
        quality_threshold: float = 1e-4,
        forcings: dict[str, torch.Tensor] | None = None,
    ) -> list[Any]:
        """Run inference with adaptive precision.

        If spectral energy tail (high-frequency noise) exceeds
        ``quality_threshold``, temporarily escalate Z1 to higher precision.

        Args:
            initial_state: initial model state.
            outer_steps: number of saved output steps.
            inner_steps: substeps between saves.
            quality_threshold: spectral tail energy ratio triggering escalation.
            forcings: optional forcing tensors.

        Returns:
            list of output states.
        """
        trajectory = []
        state = initial_state
        escalated = False

        for step in range(outer_steps):
            # Run one outer step (inner_steps substeps)
            states, _ = self.model(state, 1, inner_steps, forcings=forcings)
            state = states[-1] if isinstance(states, list) else states

            # Quality check (if profiler supports spectral audit)
            if (
                self.sda_config.profiler.spectral_audit
                and hasattr(state, "vorticity")
            ):
                self.controller.profiler.audit_spectral_energy(state.vorticity)
                tail = self.controller.profiler._spectral_tail
                if tail > quality_threshold and not escalated:
                    self.controller.hot_swap(
                        _HIGH_PRECISION_POLICY,
                        reason=f"spectral_tail={tail:.4f} > {quality_threshold}",
                    )
                    self.policy = self.controller.policy
                    escalated = True
                    logger.warning(
                        "Adaptive: escalated Z1 precision at step %d (tail=%.4f)",
                        step, tail,
                    )

            trajectory.append(state)

        return trajectory

    # ── Report ──

    def report(self) -> dict[str, Any]:
        """Return SDA session report."""
        r = self.controller.report()
        return {
            "total_steps": r.total_steps,
            "phase_transitions": r.phase_transitions,
            "profiler_summary": r.profiler_summary,
            "warnings": r.warnings,
        }

    # ── Sparse spectral inference ──

    @torch.no_grad()
    def run_sparse(
        self,
        initial_state: Any,
        outer_steps: int,
        inner_steps: int = 1,
        sparse_config: Any = None,
        forcings: dict[str, torch.Tensor] | None = None,
    ) -> tuple[list[Any], dict[str, Any]]:
        """Run inference with sparse spectral representation.

        Converts state to mixed-precision sparse between steps.
        This saves memory for long rollouts (inference / storage only).

        Args:
            initial_state: initial model state.
            outer_steps: number of saved output steps.
            inner_steps: substeps between saves.
            sparse_config: SparseConfig instance. None → default (disabled).
            forcings: optional forcing tensors.

        Returns:
            (trajectory, stats) where stats includes memory savings info.
        """
        from pytorch_src.precision.sparse.spectral_sparse import (
            SparseConfig,
            SpectralSparseState,
        )

        if sparse_config is None:
            sparse_config = SparseConfig(enabled=True, mode="mixed")

        trajectory = []
        state = initial_state
        total_dense_bytes = 0
        total_sparse_bytes = 0

        for step in range(outer_steps):
            # Run one outer step in dense mode (autograd-safe)
            states, _ = self.model(state, 1, inner_steps, forcings=forcings)
            state = states[-1] if isinstance(states, list) else states

            # Convert to sparse for storage / next step
            if sparse_config.enabled and hasattr(state, "vorticity"):
                state_dict = {
                    "vorticity": state.vorticity,
                    "divergence": state.divergence,
                    "temperature_variation": state.temperature_variation,
                    "log_surface_pressure": state.log_surface_pressure,
                }
                sparse_state = SpectralSparseState.from_state_dict(
                    state_dict, sparse_config
                )
                total_dense_bytes += sparse_state.dense_memory_bytes
                total_sparse_bytes += sparse_state.memory_bytes

                # Store the sparse state for checkpointing; recover dense
                # for next model step
                dense_dict = sparse_state.to_dense_dict()
                state = state.tree_map(
                    lambda old, name=None: dense_dict.get(name, old),
                    # fallback: keep original state unmodified
                )

            trajectory.append(state)

        savings = (
            1.0 - total_sparse_bytes / total_dense_bytes
            if total_dense_bytes > 0
            else 0.0
        )

        stats = {
            "sparse_mode": sparse_config.mode,
            "total_dense_bytes": total_dense_bytes,
            "total_sparse_bytes": total_sparse_bytes,
            "memory_savings": savings,
            "outer_steps": outer_steps,
        }
        logger.info(
            "Sparse inference: %d steps, savings=%.1f%%",
            outer_steps, savings * 100,
        )
        return trajectory, stats
