# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""Standard neural network layers — PyTorch implementation.

Zone: Z3 (BF16) — all neural network layers operate in Z3 precision.
Parameters stored in z3_param_dtype (bfloat16), master copy in z3_master_dtype
(float32) for optimizer state.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# MLP
# ═══════════════════════════════════════════════════════════════════════════

class Mlp(nn.Module):
    """Multi-layer perceptron with configurable intermediate sizes."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        intermediate_sizes: Sequence[int],
        activation: Callable = F.relu,
        activate_final: bool = False,
        use_bias: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.activate_final = activate_final

        sizes = (input_size,) + tuple(intermediate_sizes) + (output_size,)
        layers = []
        for d_in, d_out in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(d_in, d_out, bias=use_bias))
        self.layers = nn.ModuleList(layers)
        self._init_weights()

    def _init_weights(self):
        for layer in self.layers:
            # Lecun normal = kaiming_normal_ with fan_in and linear nonlinearity
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='linear')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1 or self.activate_final:
                x = self.activation(x)
        return x

    @classmethod
    def uniform(
        cls,
        input_size: int,
        output_size: int,
        *,
        hidden_size: int,
        hidden_layers: int,
        activation: Callable = F.gelu,
        activate_final: bool = False,
        use_bias: bool = True,
    ) -> Mlp:
        return cls(
            input_size, output_size,
            intermediate_sizes=(hidden_size,) * hidden_layers,
            activation=activation,
            activate_final=activate_final,
            use_bias=use_bias,
        )


class MlpUniform(Mlp):
    """MLP with uniform hidden layer sizes."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        hidden_size: int,
        n_hidden_layers: int,
        activation: Callable = F.gelu,
        activate_final: bool = False,
        use_bias: bool = True,
    ):
        super().__init__(
            input_size, output_size,
            intermediate_sizes=(hidden_size,) * n_hidden_layers,
            activation=activation,
            activate_final=activate_final,
            use_bias=use_bias,
        )


# ═══════════════════════════════════════════════════════════════════════════
# 1D Convolution (NCW / level-wise)
# ═══════════════════════════════════════════════════════════════════════════

class ConvLevel(nn.Module):
    """1D convolution in NCW format that preserves spatial dimensions."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        kernel_size: int,
        stride: int = 1,
        padding: str = "same",
        dilation: int = 1,
        use_bias: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        # PyTorch Conv1d: (N, C_in, L) -> (N, C_out, L)
        pad = kernel_size * dilation // 2 if padding == "same" else 0
        self.conv = nn.Conv1d(
            input_size, output_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            dilation=dilation,
            bias=use_bias,
        )
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='linear')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class CnnLevel(nn.Module):
    """1D CNN stack in NCW format."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        channels: Sequence[int],
        kernel_sizes: int | Sequence[int],
        strides: int | Sequence[int] = 1,
        dilations: int | Sequence[int] = 1,
        activation: Callable = F.gelu,
        activate_final: bool = False,
        use_bias: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.activate_final = activate_final

        n_total = len(channels) + 1
        all_channels = list(channels) + [output_size]
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_total
        if isinstance(dilations, int):
            dilations = [dilations] * n_total

        layers = []
        d_in = input_size
        for d_out, ks, dil in zip(all_channels, kernel_sizes, dilations):
            layers.append(ConvLevel(
                d_in, d_out, kernel_size=ks, dilation=dil, use_bias=use_bias,
            ))
            d_in = d_out
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1 or self.activate_final:
                x = self.activation(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════
# Encode-Process-Decode
# ═══════════════════════════════════════════════════════════════════════════

class Epd(nn.Module):
    """Encode-Process-Decode architecture with residual process blocks."""

    def __init__(
        self,
        encode: nn.Module,
        decode: nn.Module,
        process_blocks: Sequence[nn.Module],
        post_encode_activation: Callable | None = None,
        pre_decode_activation: Callable | None = None,
        final_activation: Callable | None = None,
    ):
        super().__init__()
        self.encode = encode
        self.decode = decode
        self.process_blocks = nn.ModuleList(process_blocks)
        self.post_encode_activation = post_encode_activation
        self.pre_decode_activation = pre_decode_activation
        self.final_activation = final_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode(x)
        if self.post_encode_activation is not None:
            h = self.post_encode_activation(h)
        for block in self.process_blocks:
            h = h + block(h)  # residual connection
        if self.pre_decode_activation is not None:
            h = self.pre_decode_activation(h)
        out = self.decode(h)
        if self.final_activation is not None:
            out = self.final_activation(out)
        return out

    @classmethod
    def build(
        cls,
        input_size: int,
        output_size: int,
        *,
        latent_size: int,
        num_process_blocks: int,
        hidden_layers: int = 2,
        hidden_size: int | None = None,
        activation: Callable = F.gelu,
    ) -> Epd:
        if hidden_size is None:
            hidden_size = latent_size
        encode = Mlp.uniform(
            input_size, latent_size,
            hidden_size=hidden_size, hidden_layers=hidden_layers,
            activation=activation,
        )
        decode = Mlp.uniform(
            latent_size, output_size,
            hidden_size=hidden_size, hidden_layers=hidden_layers,
            activation=activation,
        )
        blocks = [
            Mlp.uniform(
                latent_size, latent_size,
                hidden_size=hidden_size, hidden_layers=hidden_layers,
                activation=activation,
            )
            for _ in range(num_process_blocks)
        ]
        return cls(encode, decode, blocks, post_encode_activation=activation)


# ═══════════════════════════════════════════════════════════════════════════
# Sequential
# ═══════════════════════════════════════════════════════════════════════════

class Sequential(nn.Module):
    """Sequential layer stack."""

    def __init__(self, layers: Sequence[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════
# 2D Convolution (LonLat)
# ═══════════════════════════════════════════════════════════════════════════

class ConvLonLat(nn.Module):
    """2D convolution on (lon, lat) spatial dims.

    Padding follows the physical convention:
      - Longitude: circular/periodic (wrap-around).
      - Latitude: zero padding (no data beyond poles).

    Input: (channels, lon, lat) — no batch dim; batch is handled by towers.
    Output: (output_channels, lon, lat)

    Args:
        input_channels: number of input channels (e.g., sigma levels or features).
        output_channels: number of output channels.
        kernel_size: (lon_kernel, lat_kernel).
        use_bias: whether to include bias.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: tuple[int, int] = (3, 3),
        use_bias: bool = True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        # Compute padding extents
        self._pad_lon = (kernel_size[0] // 2, kernel_size[0] - kernel_size[0] // 2 - 1)
        self._pad_lat = (kernel_size[1] // 2, kernel_size[1] - kernel_size[1] // 2 - 1)

        self.conv = nn.Conv2d(
            input_channels, output_channels,
            kernel_size=kernel_size,
            padding=0,
            bias=use_bias,
        )
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='linear')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (channels, lon, lat)
        # Periodic padding in longitude (dim=-2)
        x = F.pad(x, (0, 0, self._pad_lon[0], self._pad_lon[1]), mode='circular')
        # Zero padding in latitude (dim=-1)
        x = F.pad(x, (self._pad_lat[0], self._pad_lat[1], 0, 0), mode='constant')
        return self.conv(x)
