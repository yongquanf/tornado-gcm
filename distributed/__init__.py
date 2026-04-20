# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""Distributed training utilities for NeuralGCM.

Provides DDP and FSDP wrappers for multi-GPU training,
plus DTensor-based tensor-parallel sharding.

This is the Phase 5 implementation (was design-decision-skip in MVP).
"""

from tornado_gcm.distributed.ddp import (
    DDPConfig,
    FSDPConfig,
    setup_distributed,
    cleanup_distributed,
    wrap_ddp,
    wrap_fsdp,
    all_reduce_metrics,
    is_main_process,
    get_world_size,
    get_rank,
    register_spectral_grad_hooks,
    get_distributed_sampler,
)

from tornado_gcm.distributed.dtensor_sharding import (
    DTensorShardingSpec,
    SpectralShardingContext,
    create_device_mesh,
    shard_tensor,
    shard_tensor_from_local,
    gather_tensor,
    redistribute_tensor,
    shard_state_dict,
    gather_state_dict,
    shard_state,
    gather_state,
    shard_module_params,
    get_neuralgcm_field_specs,
    layout_to_sharding_spec,
    maybe_enable_dtensor_sharding,
)

__all__ = [
    # DDP / FSDP
    "DDPConfig",
    "FSDPConfig",
    "setup_distributed",
    "cleanup_distributed",
    "wrap_ddp",
    "wrap_fsdp",
    "all_reduce_metrics",
    "is_main_process",
    "get_world_size",
    "get_rank",
    "register_spectral_grad_hooks",
    "get_distributed_sampler",
    # DTensor sharding
    "DTensorShardingSpec",
    "SpectralShardingContext",
    "create_device_mesh",
    "shard_tensor",
    "shard_tensor_from_local",
    "gather_tensor",
    "redistribute_tensor",
    "shard_state_dict",
    "gather_state_dict",
    "shard_state",
    "gather_state",
    "shard_module_params",
    "get_neuralgcm_field_specs",
    "layout_to_sharding_spec",
    "maybe_enable_dtensor_sharding",
]
