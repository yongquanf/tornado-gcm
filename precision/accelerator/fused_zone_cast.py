"""FusedAdvanceKernel: torch.compile-based fused zone-cast advance (O9-1).

Wraps NeuralGCMModel.step() with torch.compile to fuse the multi-step
dtype promotion chain into a single compiled graph:

    Z1(FP32) → Z3(BF16) → NN → Z1(FP32) → Z2(FP64) → Fixer → Z1(FP32)

This avoids runtime buffer allocation for intermediate dtype conversions.

Usage:
    kernel = FusedAdvanceKernel(model, model.policy)
    # Warm-up (triggers compilation)
    state = kernel(state, forcings)
    # Subsequent calls use compiled graph
    state = kernel(state, forcings)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn

from pytorch_src.precision.policy import PrecisionPolicy

logger = logging.getLogger(__name__)


class FusedAdvanceKernel:
    """Fused advance kernel using torch.compile (O9-1).

    Compiles model.step() with static dtype promotion, falling back
    to eager execution if compilation fails.

    Args:
        model: NeuralGCMModel instance.
        policy: PrecisionPolicy for zone dtype selection.
        compile_mode: torch.compile mode ('reduce-overhead', 'default',
            'max-autotune').
        dynamic: allow dynamic shapes in compiled graph.
        fallback_on_error: if True, silently fall back to eager on
            compilation failure instead of raising.
    """

    def __init__(
        self,
        model: nn.Module,
        policy: PrecisionPolicy,
        compile_mode: str = "reduce-overhead",
        dynamic: bool = False,
        fallback_on_error: bool = True,
    ) -> None:
        self._model = model
        self._policy = policy
        self._compile_mode = compile_mode
        self._dynamic = dynamic
        self._fallback_on_error = fallback_on_error
        self._compiled_step: Optional[Any] = None
        self._using_eager: bool = False
        self._call_count: int = 0

    def _ensure_compiled(self) -> None:
        """Lazily compile model.step() on first call."""
        if self._compiled_step is not None or self._using_eager:
            return

        try:
            self._compiled_step = torch.compile(
                self._model.step,
                mode=self._compile_mode,
                dynamic=self._dynamic,
                fullgraph=False,  # Allow graph breaks (zone_cast etc.)
            )
            logger.info(
                "FusedAdvanceKernel: compiled model.step() "
                "mode=%s dynamic=%s",
                self._compile_mode, self._dynamic,
            )
        except Exception as e:
            if self._fallback_on_error:
                logger.warning(
                    "FusedAdvanceKernel: compile failed (%s), "
                    "falling back to eager", e,
                )
                self._using_eager = True
            else:
                raise

    def __call__(self, state: Any, forcings: Any = None) -> Any:
        """Execute one advance step (compiled or eager).

        Args:
            state: current model State.
            forcings: optional forcing dict.

        Returns:
            Next model State.
        """
        self._ensure_compiled()
        self._call_count += 1

        if self._using_eager:
            return self._model.step(state, forcings)

        try:
            return self._compiled_step(state, forcings)
        except Exception as e:
            if self._fallback_on_error and self._call_count <= 2:
                # First-call failures are common (shape mismatch, etc.)
                logger.warning(
                    "FusedAdvanceKernel: compiled call failed (%s), "
                    "falling back to eager", e,
                )
                self._using_eager = True
                return self._model.step(state, forcings)
            raise

    def reset(self) -> None:
        """Reset compilation state (e.g. after policy hot-swap)."""
        self._compiled_step = None
        self._using_eager = False
        self._call_count = 0
        torch._dynamo.reset()
        logger.info("FusedAdvanceKernel: reset compiled state")

    @property
    def is_compiled(self) -> bool:
        """Whether the kernel is using a compiled graph."""
        return self._compiled_step is not None and not self._using_eager

    @property
    def is_eager(self) -> bool:
        """Whether the kernel fell back to eager mode."""
        return self._using_eager

    def stats(self) -> dict[str, Any]:
        """Return runtime statistics."""
        return {
            "call_count": self._call_count,
            "is_compiled": self.is_compiled,
            "is_eager": self.is_eager,
            "compile_mode": self._compile_mode,
        }
