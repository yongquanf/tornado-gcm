# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""Streaming normalization (StreamNorm) — PyTorch implementation.

This replaces the JAX StreamNorm from NeuralGCM which maintains running
statistics (mean, variance) for online normalization without storing full data.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StreamNorm(nn.Module):
    """Streaming normalization using exponential moving average statistics.

    Maintains running mean and variance that update during training and are
    frozen during evaluation.

    Args:
        num_features: number of features to normalize.
        momentum: EMA decay factor (0 = no update, 1 = replace).
        eps: epsilon for numerical stability.
    """

    def __init__(
        self,
        num_features: int,
        momentum: float = 0.01,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize x using running statistics.

        x: (..., num_features)
        """
        if self.training:
            # Compute batch statistics over all dims except last
            dims = list(range(x.ndim - 1))
            batch_mean = x.detach().mean(dim=dims)
            batch_var = x.detach().var(dim=dims, unbiased=False)
            m = self.momentum
            self.running_mean.mul_(1 - m).add_(m * batch_mean)
            self.running_var.mul_(1 - m).add_(m * batch_var)
            self.num_batches_tracked += 1

        return (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse the normalization: x_orig = x * std + mean."""
        std = torch.sqrt(self.running_var + self.eps)
        return x * std + self.running_mean
