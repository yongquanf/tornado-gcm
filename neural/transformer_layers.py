# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""Transformer layers — PyTorch implementation.

Zone: Z3 (BF16) — attention and NN computation in bfloat16.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# Multi-Head Attention
# ═══════════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional QK normalization.

    Supports self-attention and cross-attention.

    Args:
        num_heads: number of attention heads.
        in_features: input feature dimension.
        qkv_features: total QKV dimension (default: in_features).
        out_features: output dimension (default: in_features).
        in_kv_features: KV input dimension for cross-attention (default: in_features).
        normalize_qk: apply LayerNorm to Q and K (ViT-22B stabilization).
        use_bias: use bias in projections.
    """

    def __init__(
        self,
        num_heads: int,
        in_features: int,
        qkv_features: int | None = None,
        out_features: int | None = None,
        in_kv_features: int | None = None,
        *,
        normalize_qk: bool = False,
        use_bias: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.in_features = in_features
        self.qkv_features = qkv_features or in_features
        self.out_features = out_features or in_features
        self.in_kv_features = in_kv_features or in_features

        if self.qkv_features % num_heads != 0:
            raise ValueError(
                f"qkv_features ({self.qkv_features}) must be divisible "
                f"by num_heads ({num_heads})"
            )
        self.head_dim = self.qkv_features // num_heads

        self.q_proj = nn.Linear(in_features, self.qkv_features, bias=use_bias)
        self.k_proj = nn.Linear(self.in_kv_features, self.qkv_features, bias=use_bias)
        self.v_proj = nn.Linear(self.in_kv_features, self.qkv_features, bias=use_bias)
        self.out_proj = nn.Linear(self.qkv_features, self.out_features, bias=use_bias)

        self.normalize_qk = normalize_qk
        if normalize_qk:
            self.q_ln = nn.LayerNorm(self.head_dim)
            self.k_ln = nn.LayerNorm(self.head_dim)

        self._init_weights()

    def _init_weights(self):
        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            query: (batch..., seq_q, d_in)
            key: (batch..., seq_kv, d_kv) — defaults to query for self-attention
            value: (batch..., seq_kv, d_kv) — defaults to key
            attention_bias: additive bias (batch..., heads, seq_q, seq_kv)
            mask: boolean mask (True = attend) (batch..., seq_q, seq_kv)

        Returns:
            (batch..., seq_q, d_out)
        """
        if key is None:
            key = query
        if value is None:
            value = key

        # Project
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape to (batch..., seq, heads, head_dim)
        batch_shape = q.shape[:-1]
        seq_q = q.shape[-2]
        seq_kv = k.shape[-2]

        q = q.view(*batch_shape[:-1], seq_q, self.num_heads, self.head_dim)
        k = k.view(*batch_shape[:-1], seq_kv, self.num_heads, self.head_dim)
        v = v.view(*batch_shape[:-1], seq_kv, self.num_heads, self.head_dim)

        if self.normalize_qk:
            q = self.q_ln(q)
            k = self.k_ln(k)

        # Transpose to (batch..., heads, seq, head_dim)
        q = q.transpose(-3, -2)
        k = k.transpose(-3, -2)
        v = v.transpose(-3, -2)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale

        if attention_bias is not None:
            attn = attn + attention_bias
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(-3), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        # Transpose back and reshape
        out = out.transpose(-3, -2).contiguous()
        out = out.view(*batch_shape[:-1], seq_q, self.qkv_features)
        return self.out_proj(out)


# ═══════════════════════════════════════════════════════════════════════════
# Transformer Block
# ═══════════════════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with self-attention and optional cross-attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int | None = None,
        activation: Callable = F.gelu,
        normalize_qk: bool = False,
        cross_attention: bool = False,
        d_cross: int | None = None,
        use_bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model

        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(
            num_heads, d_model, normalize_qk=normalize_qk, use_bias=use_bias,
        )

        self.cross_attention = cross_attention
        if cross_attention:
            self.ln_cross = nn.LayerNorm(d_model)
            self.cross_attn = MultiHeadAttention(
                num_heads, d_model, in_kv_features=d_cross or d_model,
                normalize_qk=normalize_qk, use_bias=use_bias,
            )

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=use_bias),
            nn.GELU() if activation is F.gelu else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=use_bias),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        cross_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Self-attention
        h = self.ln1(x)
        h = self.self_attn(h, mask=mask)
        x = x + self.dropout(h)

        # Cross-attention
        if self.cross_attention and context is not None:
            h = self.ln_cross(x)
            h = self.cross_attn(h, context, context, mask=cross_mask)
            x = x + self.dropout(h)

        # FFN
        h = self.ln2(x)
        x = x + self.ffn(h)
        return x


# ═══════════════════════════════════════════════════════════════════════════
# Transformer Encoder / Decoder
# ═══════════════════════════════════════════════════════════════════════════

class TransformerEncoder(nn.Module):
    """Stack of Transformer blocks (self-attention only)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int | None = None,
        activation: Callable = F.gelu,
        normalize_qk: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model, num_heads, d_ff=d_ff,
                activation=activation, normalize_qk=normalize_qk,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.final_ln(x)


class TransformerDecoder(nn.Module):
    """Stack of Transformer blocks with cross-attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_cross: int | None = None,
        d_ff: int | None = None,
        activation: Callable = F.gelu,
        normalize_qk: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model, num_heads, d_ff=d_ff,
                activation=activation, normalize_qk=normalize_qk,
                cross_attention=True, d_cross=d_cross,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask: torch.Tensor | None = None,
        cross_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, context=context, mask=mask, cross_mask=cross_mask)
        return self.final_ln(x)


# ═══════════════════════════════════════════════════════════════════════════
# Positional encoding
# ═══════════════════════════════════════════════════════════════════════════

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to x of shape (..., seq_len, d_model)."""
        seq_len = x.shape[-2]
        return x + self.pe[..., :seq_len, :]
