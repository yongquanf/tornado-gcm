# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""ZoneCast: dtype conversion utilities for PZHA zone boundaries.

Provides:
    - zone_cast(): explicit dtype conversion with optional precision audit
    - @zone_boundary decorator: auto-cast inputs/outputs at zone transitions
    - @f64_math decorator: run a function in FP64 (Z0/Z2) context
    - einsum_highest(): einsum with FP64 intermediate precision (Z0/Z1)
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Optional, TypeVar

import torch

from tornado_gcm.precision.policy import PrecisionPolicy, PrecisionZone

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable)


# ---------------------------------------------------------------------------
# Helpers: cast a single arg that may be Tensor, pytree, or passthrough
# ---------------------------------------------------------------------------

def _cast_arg_zone(arg: Any, src: PrecisionZone, dst: PrecisionZone,
                   policy: PrecisionPolicy, **kw) -> Any:
    """Cast a single argument — Tensor, tree_map-able object, or passthrough."""
    if isinstance(arg, torch.Tensor):
        return zone_cast(arg, src, dst, policy, **kw)
    if hasattr(arg, "tree_map"):          # State / dataclass with tree_map
        return tree_zone_cast(arg, src, dst, policy, **kw)
    return arg


def _cast_arg_dtype(arg: Any, dtype: torch.dtype) -> Any:
    """Cast a single argument to *dtype* — Tensor, tree_map-able, or passthrough."""
    if isinstance(arg, torch.Tensor):
        return arg.to(dtype)
    if hasattr(arg, "tree_map"):
        return arg.tree_map(lambda t: t.to(dtype) if isinstance(t, torch.Tensor) else t)
    return arg


def _cast_result_dtype(result: Any, dtype: torch.dtype) -> Any:
    """Cast function output (Tensor, tuple, or tree_map-able) to *dtype*."""
    if isinstance(result, torch.Tensor):
        return result.to(dtype)
    if isinstance(result, tuple):
        return tuple(
            r.to(dtype) if isinstance(r, torch.Tensor)
            else (r.tree_map(lambda t: t.to(dtype) if isinstance(t, torch.Tensor) else t)
                  if hasattr(r, "tree_map") else r)
            for r in result
        )
    if hasattr(result, "tree_map"):
        return result.tree_map(lambda t: t.to(dtype) if isinstance(t, torch.Tensor) else t)
    return result


# ---------------------------------------------------------------------------
# Precision rank ordering (for automatic up/down-cast decisions)
# ---------------------------------------------------------------------------
_DTYPE_RANK = {
    torch.bfloat16: 0,
    torch.float16: 0,
    torch.float32: 1,
    torch.float64: 2,
}


def _dtype_rank(dtype: torch.dtype) -> int:
    return _DTYPE_RANK.get(dtype, 1)


# ═══════════════════════════════════════════════════════════════════════════
# Core zone_cast function
# ═══════════════════════════════════════════════════════════════════════════

def zone_cast(
    x: torch.Tensor,
    src_zone: PrecisionZone,
    dst_zone: PrecisionZone,
    policy: PrecisionPolicy,
    *,
    monitor: Optional[Any] = None,
    name: str = "",
) -> torch.Tensor:
    """Cast tensor from source zone precision to destination zone precision.

    Implements Algorithm 1 (ZoneCast) from 混合精度加速完整方案.tex:
    - Same zone: no-op
    - Upcast (to higher precision): lossless promotion
    - Downcast (to lower precision): cast + optional audit

    Args:
        x: Input tensor.
        src_zone: Source precision zone.
        dst_zone: Destination precision zone.
        policy: PrecisionPolicy defining zone dtypes.
        monitor: Optional PrecisionMonitor for audit logging.
        name: Human-readable name for audit logging.

    Returns:
        Tensor cast to destination zone's compute dtype.
    """
    if src_zone == dst_zone:
        return x

    src_dtype = policy.compute_dtype(src_zone)
    dst_dtype = policy.compute_dtype(dst_zone)

    if x.dtype == dst_dtype:
        return x

    x_cast = x.to(dst_dtype)

    # Audit precision loss on downcast
    if (
        _dtype_rank(dst_dtype) < _dtype_rank(src_dtype)
        and monitor is not None
        and policy.enable_precision_audit
    ):
        monitor.audit_zone_transfer(
            name or f"{src_zone.value}→{dst_zone.value}", x, x_cast
        )

    return x_cast


# ═══════════════════════════════════════════════════════════════════════════
# Tree-level zone_cast (operates on dict / dataclass fields)
# ═══════════════════════════════════════════════════════════════════════════

def tree_zone_cast(
    tree: Any,
    src_zone: PrecisionZone,
    dst_zone: PrecisionZone,
    policy: PrecisionPolicy,
    **kwargs,
) -> Any:
    """Apply zone_cast to all tensors in a pytree-like structure."""
    if isinstance(tree, torch.Tensor):
        return zone_cast(tree, src_zone, dst_zone, policy, **kwargs)
    elif isinstance(tree, dict):
        return {
            k: tree_zone_cast(v, src_zone, dst_zone, policy, **kwargs)
            for k, v in tree.items()
        }
    elif isinstance(tree, (list, tuple)):
        cast_items = [
            tree_zone_cast(v, src_zone, dst_zone, policy, **kwargs)
            for v in tree
        ]
        return type(tree)(cast_items)
    elif hasattr(tree, "tree_map"):
        # Supports State/dataclass with tree_map method
        return tree.tree_map(
            lambda t: zone_cast(t, src_zone, dst_zone, policy, **kwargs)
            if isinstance(t, torch.Tensor)
            else t
        )
    else:
        return tree


# ═══════════════════════════════════════════════════════════════════════════
# @zone_boundary decorator
# ═══════════════════════════════════════════════════════════════════════════

def zone_boundary(
    src: PrecisionZone,
    dst: PrecisionZone,
    policy: PrecisionPolicy,
    *,
    cast_output: bool = True,
) -> Callable[[F], F]:
    """Decorator that casts tensor inputs to `dst` zone and outputs back if needed.

    Usage::

        @zone_boundary(PrecisionZone.Z1_DYNAMICS_CORE, PrecisionZone.Z3_NEURAL_NETWORK, policy)
        def nn_forward(x: Tensor) -> Tensor:
            ...  # x is already in Z3 (BF16), output cast back to Z1 (FP32)
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # Cast positional args to dst zone (Tensor, State, or passthrough)
            cast_args = tuple(
                _cast_arg_zone(a, src, dst, policy) for a in args
            )
            cast_kwargs = {
                k: _cast_arg_zone(v, src, dst, policy)
                for k, v in kwargs.items()
            }
            result = fn(*cast_args, **cast_kwargs)
            if cast_output:
                result = _cast_arg_zone(result, dst, src, policy)
            return result

        return wrapper  # type: ignore[return-value]

    return decorator


# ═══════════════════════════════════════════════════════════════════════════
# @f64_math decorator  (Z0 / Z2 context)
# ═══════════════════════════════════════════════════════════════════════════

def f64_math(fn: F) -> F:
    """Decorator: execute function with all Tensor args promoted to float64.

    Equivalent to JAX's ``_with_f64_math`` pattern used in sigma_coordinates.
    Output tensors are cast back to float32.

    Supports Tensor args, State/dataclass args (via tree_map), and passthrough.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        args64 = tuple(_cast_arg_dtype(a, torch.float64) for a in args)
        kwargs64 = {
            k: _cast_arg_dtype(v, torch.float64) for k, v in kwargs.items()
        }
        result = fn(*args64, **kwargs64)
        return _cast_result_dtype(result, torch.float32)

    return wrapper  # type: ignore[return-value]


# ═══════════════════════════════════════════════════════════════════════════
# einsum_highest: FP64 intermediate precision einsum
# ═══════════════════════════════════════════════════════════════════════════

def einsum_highest(subscripts: str, *operands: torch.Tensor) -> torch.Tensor:
    """Einsum with FP64 intermediate precision, output in highest input dtype.

    Equivalent to ``jnp.einsum(..., precision=lax.Precision.HIGHEST)``.
    Critical for numerical stability in spherical harmonic transforms (Z1)
    and implicit integration matrices (Z0).
    """
    if not operands:
        raise ValueError("einsum_highest requires at least one operand")

    # Match JAX behavior more closely: compute in FP64, then return in the
    # highest precision among input floating dtypes (instead of hard float32).
    # This avoids silently dropping implicit-coupling precision.
    float_ops = [op for op in operands if op.is_floating_point()]
    if not float_ops:
        return torch.einsum(subscripts, *operands)

    out_dtype = max(float_ops, key=lambda t: _dtype_rank(t.dtype)).dtype
    ops64 = [op.to(torch.float64) for op in operands]
    result = torch.einsum(subscripts, *ops64)
    return result.to(out_dtype)
