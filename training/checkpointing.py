# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint utilities for NeuralGCM.

Saves/loads model + optimizer + EMA parameters + training state.
Supports converting NN params to BF16 for inference storage.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

import torch
import torch.nn as nn


@dataclasses.dataclass
class TrainingState:
    """Serializable training state."""

    step: int = 0
    phase: int = 0
    best_val_loss: float = float("inf")


def save_checkpoint(
    path: str | pathlib.Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    training_state: TrainingState,
    ema_params: dict[str, torch.Tensor] | None = None,
) -> None:
    """Save a training checkpoint."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_state": dataclasses.asdict(training_state),
    }
    if ema_params is not None:
        checkpoint["ema_params"] = ema_params
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | pathlib.Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> TrainingState:
    """Load a training checkpoint.

    Returns the TrainingState. Model and optimizer are updated in-place.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    ts = checkpoint.get("training_state", {})
    return TrainingState(**ts)


def export_inference_checkpoint(
    model: nn.Module,
    path: str | pathlib.Path,
    ema_params: dict[str, torch.Tensor] | None = None,
    nn_dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Export a checkpoint for inference with NN params in BF16.

    Args:
        model: trained model.
        path: output path.
        ema_params: if provided, use EMA weights instead of raw params.
        nn_dtype: target dtype for NN parameters (default BF16).
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if ema_params is not None:
        state_dict = dict(ema_params)
    else:
        state_dict = model.state_dict()

    # Convert NN parameters to target dtype
    converted = {}
    for name, param in state_dict.items():
        if param.is_floating_point():
            converted[name] = param.to(nn_dtype)
        else:
            converted[name] = param

    torch.save({"model_state_dict": converted, "format": "inference"}, path)
