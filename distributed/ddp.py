# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""DDP and FSDP wrappers for NeuralGCM distributed training.

Implements:
  - DDPConfig / FSDPConfig: dataclass configurations
  - setup_distributed / cleanup_distributed: process group lifecycle
  - wrap_ddp / wrap_fsdp: model wrapping with precision-aware sharding
  - all_reduce_metrics: synchronized metric aggregation
  - Spectral-aware gradient synchronization hooks

This replaces JAX's SPMD/pjit/shard_map with PyTorch-native DDP/FSDP.
The spectral state (modal coefficients) needs special treatment:
  - Gradient all-reduce must preserve FP32 precision for dynamics (Z1)
  - FSDP sharding can split along the level axis (not spectral axes)
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import os
from typing import Any, Callable, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean flag from the environment."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass
class DDPConfig:
    """Configuration for DistributedDataParallel.

    Attributes:
        backend: distributed backend ('nccl', 'gloo').
        find_unused_parameters: DDP option for dynamic graphs.
        gradient_as_bucket_view: memory optimization.
        static_graph: enable static graph optimization.
        broadcast_buffers: sync batch-norm buffers.
        bucket_cap_mb: gradient bucket size in MB.
    """
    backend: str = "nccl"
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    static_graph: bool = True
    broadcast_buffers: bool = True
    bucket_cap_mb: int = 25


@dataclasses.dataclass
class FSDPConfig:
    """Configuration for FullyShardedDataParallel.

    Attributes:
        sharding_strategy: FSDP sharding level.
            'FULL_SHARD': maximum memory savings, all-gather per forward.
            'SHARD_GRAD_OP': shard only optimizer states and gradients.
            'NO_SHARD': equivalent to DDP (for debugging).
        auto_wrap_policy: how to wrap sub-modules.
            'size_based': wrap modules exceeding min_num_params.
            'transformer': wrap Transformer blocks individually.
        min_num_params: threshold for size_based wrapping.
        mixed_precision: enable FSDP mixed precision.
        cpu_offload: offload parameters to CPU (saves GPU memory).
        forward_prefetch: prefetch next all-gather during forward.
        backward_prefetch: prefetch during backward pass.
        limit_all_gathers: throttle all-gathers to prevent OOM.
        dtensor_aware: whether the run was launched through the DTensor-aware path.
    """
    sharding_strategy: str = "FULL_SHARD"
    auto_wrap_policy: str = "size_based"
    min_num_params: int = 100_000
    mixed_precision: bool = True
    cpu_offload: bool = False
    forward_prefetch: bool = True
    backward_prefetch: str = "BACKWARD_PRE"
    limit_all_gathers: bool = True
    dtensor_aware: bool = False
    dtensor_mesh_shape: str = ""
    dtensor_mesh_dim_names: str = ""
    dtensor_param_layout: str = "replicate"
    dtensor_state_layout: str = "level"
    dtensor_config_path: str = ""

    @classmethod
    def from_env(cls) -> "FSDPConfig":
        """Build an FSDP config from environment overrides."""
        return cls(
            sharding_strategy=os.environ.get("NEURALGCM_FSDP_SHARDING_STRATEGY", "FULL_SHARD"),
            auto_wrap_policy=os.environ.get("NEURALGCM_FSDP_AUTO_WRAP_POLICY", "size_based"),
            min_num_params=int(os.environ.get("NEURALGCM_FSDP_MIN_NUM_PARAMS", "100000")),
            mixed_precision=_env_flag("NEURALGCM_FSDP_MIXED_PRECISION", True),
            cpu_offload=_env_flag("NEURALGCM_FSDP_CPU_OFFLOAD", False),
            forward_prefetch=_env_flag("NEURALGCM_FSDP_FORWARD_PREFETCH", True),
            backward_prefetch=os.environ.get("NEURALGCM_FSDP_BACKWARD_PREFETCH", "BACKWARD_PRE"),
            limit_all_gathers=_env_flag("NEURALGCM_FSDP_LIMIT_ALL_GATHERS", True),
            dtensor_aware=_env_flag("NEURALGCM_DTENSOR_ENABLED", False),
            dtensor_mesh_shape=os.environ.get("NEURALGCM_DTENSOR_MESH_SHAPE", ""),
            dtensor_mesh_dim_names=os.environ.get("NEURALGCM_DTENSOR_MESH_DIM_NAMES", ""),
            dtensor_param_layout=os.environ.get("NEURALGCM_DTENSOR_PARAM_LAYOUT", "replicate"),
            dtensor_state_layout=os.environ.get("NEURALGCM_DTENSOR_STATE_LAYOUT", "level"),
            dtensor_config_path=os.environ.get("NEURALGCM_DTENSOR_CONFIG_PATH", ""),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Process group setup / cleanup
# ═══════════════════════════════════════════════════════════════════════════


def setup_distributed(
    backend: str = "nccl",
    init_method: str | None = None,
    world_size: int | None = None,
    rank: int | None = None,
) -> None:
    """Initialize the distributed process group.

    Reads RANK, WORLD_SIZE, LOCAL_RANK from environment if not provided
    (compatible with torchrun / torch.distributed.launch).

    Args:
        backend: 'nccl' for GPU, 'gloo' for CPU.
        init_method: URL for rendezvous (default: env://).
        world_size: total number of processes.
        rank: global rank of this process.
    """
    if dist.is_initialized():
        return

    rank = rank or int(os.environ.get("RANK", 0))
    world_size = world_size or int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if init_method is None:
        init_method = "env://"

    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    logger.info(
        f"Distributed initialized: rank={rank}/{world_size}, "
        f"local_rank={local_rank}, backend={backend}"
    )


def cleanup_distributed() -> None:
    """Destroy the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if this is the main (rank 0) process."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size() -> int:
    """Get the total number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """Get the global rank of this process."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


# ═══════════════════════════════════════════════════════════════════════════
# DDP wrapper
# ═══════════════════════════════════════════════════════════════════════════


def wrap_ddp(
    model: nn.Module,
    config: DDPConfig | None = None,
    device_id: int | None = None,
) -> DDP:
    """Wrap a model with DistributedDataParallel.

    Handles device placement and precision-aware gradient bucketing.

    Args:
        model: the model to distribute.
        config: DDP configuration (uses defaults if None).
        device_id: GPU device for this process (auto-detects from LOCAL_RANK).

    Returns:
        DDP-wrapped model.
    """
    if config is None:
        config = DDPConfig()

    if device_id is None:
        device_id = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        model = model.to(f"cuda:{device_id}")

    return DDP(
        model,
        device_ids=[device_id] if torch.cuda.is_available() else None,
        find_unused_parameters=config.find_unused_parameters,
        gradient_as_bucket_view=config.gradient_as_bucket_view,
        static_graph=config.static_graph,
        broadcast_buffers=config.broadcast_buffers,
        bucket_cap_mb=config.bucket_cap_mb,
    )


# ═══════════════════════════════════════════════════════════════════════════
# FSDP wrapper
# ═══════════════════════════════════════════════════════════════════════════


def wrap_fsdp(
    model: nn.Module,
    config: FSDPConfig | None = None,
) -> nn.Module:
    """Wrap a model with FullyShardedDataParallel.

    Uses PZHA-aware mixed precision:
      - Parameters: float32 (master weights)
      - Compute: bfloat16 for Z3 (NN layers)
      - Reduce: float32 for gradient all-reduce

    Args:
        model: the model to shard.
        config: FSDP configuration (uses defaults if None).

    Returns:
        FSDP-wrapped model.
    """
    try:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            ShardingStrategy,
            MixedPrecision,
            CPUOffload,
            BackwardPrefetch,
        )
        from torch.distributed.fsdp.wrap import (
            size_based_auto_wrap_policy,
            transformer_auto_wrap_policy,
        )
    except ImportError:
        raise ImportError(
            "FSDP requires PyTorch >= 1.12. Please upgrade."
        )

    if config is None:
        config = FSDPConfig.from_env()

    if config.dtensor_aware:
        logger.info(
            "DTensor-aware FSDP config requested: mesh=%s dims=%s param_layout=%s state_layout=%s config_path=%s",
            config.dtensor_mesh_shape or "auto",
            config.dtensor_mesh_dim_names or "auto",
            config.dtensor_param_layout,
            config.dtensor_state_layout,
            config.dtensor_config_path or "<none>",
        )
        from tornado_gcm.distributed.dtensor_sharding import maybe_enable_dtensor_sharding
        model = maybe_enable_dtensor_sharding(model, config)

    # Sharding strategy
    strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
    }
    sharding_strategy = strategy_map.get(
        config.sharding_strategy, ShardingStrategy.FULL_SHARD
    )

    # Mixed precision (PZHA-aware)
    mp = None
    if config.mixed_precision:
        mp = MixedPrecision(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )

    # CPU offload
    cpu_offload = CPUOffload(offload_params=config.cpu_offload)

    # Backward prefetch
    prefetch_map = {
        "BACKWARD_PRE": BackwardPrefetch.BACKWARD_PRE,
        "BACKWARD_POST": BackwardPrefetch.BACKWARD_POST,
    }
    backward_prefetch = prefetch_map.get(
        config.backward_prefetch, BackwardPrefetch.BACKWARD_PRE
    )

    # Auto-wrap policy
    wrap_policy = None
    if config.auto_wrap_policy == "size_based":
        wrap_policy = size_based_auto_wrap_policy
    elif config.auto_wrap_policy == "transformer":
        from tornado_gcm.neural.transformer_layers import TransformerEncoderBlock
        wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={TransformerEncoderBlock},
        )

    wrapped = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        mixed_precision=mp,
        cpu_offload=cpu_offload,
        backward_prefetch=backward_prefetch,
        auto_wrap_policy=wrap_policy,
        forward_prefetch=config.forward_prefetch,
        limit_all_gathers=config.limit_all_gathers,
    )

    for attr in (
        "_dtensor_mesh",
        "_dtensor_field_specs",
        "_dtensor_state_default_spec",
        "_dtensor_param_default_spec",
        "_dtensor_spectral_context",
        "_dtensor_parameterization_sharded",
        "shard_state_with_dtensor",
        "gather_state_from_dtensor",
    ):
        if hasattr(model, attr):
            setattr(wrapped, attr, getattr(model, attr))

    return wrapped


# ═══════════════════════════════════════════════════════════════════════════
# Metric aggregation
# ═══════════════════════════════════════════════════════════════════════════


def all_reduce_metrics(
    metrics: dict[str, float | torch.Tensor],
    op: str = "mean",
) -> dict[str, float]:
    """Synchronize and aggregate metrics across all processes.

    Args:
        metrics: dictionary of metric name → value.
        op: aggregation operation ('mean', 'sum', 'max').

    Returns:
        Aggregated metrics dict (float values).
    """
    if not dist.is_initialized() or get_world_size() == 1:
        return {k: float(v) if isinstance(v, torch.Tensor) else v
                for k, v in metrics.items()}

    reduce_op = {
        "mean": dist.ReduceOp.SUM,
        "sum": dist.ReduceOp.SUM,
        "max": dist.ReduceOp.MAX,
    }[op]

    result = {}
    for key, val in metrics.items():
        if isinstance(val, torch.Tensor):
            t = val.clone().detach()
        else:
            t = torch.tensor(float(val))

        if torch.cuda.is_available():
            t = t.cuda()

        dist.all_reduce(t, op=reduce_op)

        if op == "mean":
            t = t / get_world_size()

        result[key] = t.item()

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Spectral-aware gradient hooks
# ═══════════════════════════════════════════════════════════════════════════


def register_spectral_grad_hooks(
    model: nn.Module,
    spectral_param_patterns: Sequence[str] = ("dynamics",),
) -> list:
    """Register gradient hooks to ensure spectral parameters use FP32 all-reduce.

    In standard DDP with mixed precision, gradients may be reduced in BF16.
    This hook forces spectral-sensitive parameters (dynamics core) to use
    FP32 all-reduce, preserving numerical stability.

    Args:
        model: model (can be DDP-wrapped or raw).
        spectral_param_patterns: name patterns for spectral parameters.

    Returns:
        list of hook handles (for removal if needed).
    """
    handles = []
    import re

    patterns = [re.compile(p) for p in spectral_param_patterns]

    for name, param in model.named_parameters():
        if any(p.search(name) for p in patterns):
            def _hook(grad, param_name=name):
                if grad is not None and grad.dtype != torch.float32:
                    return grad.float()
                return grad
            h = param.register_hook(_hook)
            handles.append(h)
            logger.debug(f"Registered FP32 grad hook for {name}")

    return handles


# ═══════════════════════════════════════════════════════════════════════════
# Distributed sampler helper
# ═══════════════════════════════════════════════════════════════════════════


def get_distributed_sampler(
    dataset: torch.utils.data.Dataset,
    shuffle: bool = True,
    seed: int = 42,
) -> torch.utils.data.DistributedSampler | None:
    """Create a DistributedSampler if running in distributed mode.

    Returns None for single-process runs, enabling transparent fallback.
    """
    if not dist.is_initialized() or get_world_size() == 1:
        return None

    return torch.utils.data.DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=shuffle,
        seed=seed,
    )
