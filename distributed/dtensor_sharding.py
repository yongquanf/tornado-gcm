# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""DTensor-based tensor parallel sharding for NeuralGCM.

Uses ``torch.distributed.tensor`` (DTensor) to shard simulation tensors
along specified dimensions with *transparent* collective communication.

Design rationale
────────────────
NeuralGCM tensors have the general shapes:

  nodal fields : (n_layers, lon_nodes, lat_nodes)
  modal fields : (n_layers, 2*M-1, L)
  batched      : (batch, n_layers, ...)

The spectral transform (SHT) globally couples the spatial/wavenumber axes,
so the natural shard axis is:

  * **level (dim=0)** — vertical layers are nearly independent between
    time-steps; sharding here avoids any redistribution during SHT.
  * **batch (dim=0)** — trivially parallel when a batch dimension exists.

Sharding along lon or lat is possible for nodal-only work but requires
``redistribute`` to Replicate before every SHT call, which is typically
not worth the communication cost.

Public API
──────────
  create_device_mesh   — Build a 1-D or 2-D DeviceMesh for NeuralGCM.
  DTensorShardingSpec  — Declares per-field shard dimensions.
  shard_tensor         — Convert a plain Tensor into a DTensor.
  gather_tensor        — Collect a DTensor back to a full Tensor.
  shard_state_dict     — Shard every Tensor in a state dict.
  gather_state_dict    — Gather every DTensor in a state dict.
  shard_module_params  — Shard nn.Module parameters in-place.

Requires PyTorch ≥ 2.1 (DTensor graduated from prototype in 2.1).
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Dict, Optional, Sequence, Union

import torch
import torch.nn as nn

from tornado_gcm.core.primitive_equations import State

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class LocalDeviceMesh:
    """Single-rank fallback mesh for local development environments."""
    mesh_shape: tuple[int, ...]
    mesh_dim_names: tuple[str, ...]
    device_type: str = "cpu"

    @property
    def ndim(self) -> int:
        return len(self.mesh_shape)

    @property
    def mesh(self) -> torch.Tensor:
        return torch.zeros(self.mesh_shape, dtype=torch.int64)


# ── lazy imports (fail clearly if DTensor unavailable) ─────────────────

_DTENSOR_AVAILABLE: bool | None = None


def _check_dtensor() -> bool:
    global _DTENSOR_AVAILABLE
    if _DTENSOR_AVAILABLE is None:
        try:
            from torch.distributed.tensor import DTensor, Shard, Replicate  # noqa: F401
            from torch.distributed.device_mesh import DeviceMesh  # noqa: F401
            _DTENSOR_AVAILABLE = True
        except ImportError:
            _DTENSOR_AVAILABLE = False
    return _DTENSOR_AVAILABLE


def _require_dtensor():
    if not _check_dtensor():
        raise ImportError(
            "torch.distributed.tensor is not available. "
            "DTensor requires PyTorch >= 2.1.  "
            "Install with: pip install torch>=2.1"
        )


def _parse_mesh_shape(mesh_shape: str | tuple[int, ...] | None) -> tuple[int, ...] | None:
    """Parse a DTensor mesh shape from config text or tuple."""
    if mesh_shape is None or mesh_shape == "":
        return None
    if isinstance(mesh_shape, tuple):
        return mesh_shape
    raw = str(mesh_shape).strip().lower()
    if raw in {"", "auto", "none"}:
        return None
    parts = tuple(int(p.strip()) for p in raw.split(",") if p.strip())
    if not parts or any(p <= 0 for p in parts):
        raise ValueError(f"Invalid DTensor mesh shape: {mesh_shape}")
    return parts


def _parse_mesh_dim_names(
    mesh_dim_names: str | tuple[str, ...] | None,
    ndim: int,
) -> tuple[str, ...] | None:
    """Parse logical mesh-dimension names."""
    if mesh_dim_names is None or mesh_dim_names == "":
        return None
    if isinstance(mesh_dim_names, tuple):
        return mesh_dim_names
    parts = tuple(p.strip() for p in str(mesh_dim_names).split(",") if p.strip())
    if len(parts) != ndim:
        raise ValueError(
            f"Mesh dim names {parts} do not match mesh ndim={ndim}"
        )
    return parts


# ═════════════════════════════════════════════════════════════════════════
# DeviceMesh creation
# ═════════════════════════════════════════════════════════════════════════


def create_device_mesh(
    mesh_shape: tuple[int, ...] | None = None,
    mesh_dim_names: tuple[str, ...] | None = None,
    device_type: str = "cuda",
) -> "torch.distributed.device_mesh.DeviceMesh":
    """Create a DeviceMesh for NeuralGCM parallel training / inference.

    Common configurations::

      # 1-D data-parallel (4 GPUs)
      mesh = create_device_mesh((4,), ("dp",))

      # 2-D: 2 data-parallel × 2 tensor-parallel
      mesh = create_device_mesh((2, 2), ("dp", "tp"))

    If *mesh_shape* is ``None`` it falls back to ``(world_size,)`` with
    dim name ``"dp"``.

    Args:
        mesh_shape: shape of the process mesh (e.g. ``(4,)`` or ``(2, 2)``).
        mesh_dim_names: human-readable names for each mesh dimension.
        device_type: ``"cuda"`` or ``"cpu"``.

    Returns:
        A ``DeviceMesh`` instance.
    """
    _require_dtensor()
    from torch.distributed.device_mesh import init_device_mesh

    import torch.distributed as dist
    if not dist.is_initialized():
        raise RuntimeError(
            "Distributed process group must be initialized before "
            "creating a DeviceMesh.  Call setup_distributed() first."
        )

    world_size = dist.get_world_size()

    if mesh_shape is None:
        mesh_shape = (world_size,)
        mesh_dim_names = ("dp",)
    if mesh_dim_names is None:
        mesh_dim_names = tuple(f"dim{i}" for i in range(len(mesh_shape)))

    prod = 1
    for s in mesh_shape:
        prod *= s
    if prod != world_size:
        raise ValueError(
            f"mesh_shape {mesh_shape} (product={prod}) does not match "
            f"world_size={world_size}"
        )

    mesh_kwargs = {"mesh_dim_names": mesh_dim_names}
    backend = dist.get_backend()
    if backend:
        mesh_kwargs["backend_override"] = backend

    try:
        try:
            mesh = init_device_mesh(
                device_type,
                mesh_shape,
                **mesh_kwargs,
            )
        except TypeError:
            mesh_kwargs.pop("backend_override", None)
            mesh = init_device_mesh(
                device_type,
                mesh_shape,
                **mesh_kwargs,
            )
        logger.info(
            f"DeviceMesh created: shape={mesh_shape}, "
            f"names={mesh_dim_names}, device={device_type}"
        )
        return mesh
    except Exception as exc:
        if world_size == 1:
            fallback_names = mesh_dim_names or tuple(f"dim{i}" for i in range(len(mesh_shape)))
            logger.warning(
                "Falling back to LocalDeviceMesh for single-rank development mode: %s",
                exc,
            )
            return LocalDeviceMesh(
                mesh_shape=tuple(mesh_shape),
                mesh_dim_names=tuple(fallback_names),
                device_type=device_type,
            )
        raise


# ═════════════════════════════════════════════════════════════════════════
# Sharding specification
# ═════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass
class DTensorShardingSpec:
    """Describes how to shard NeuralGCM tensors on a DeviceMesh.

    For a 1-D mesh the *placements* list has one element; for N-D mesh it
    has N elements.

    Pre-built factory methods cover the common NeuralGCM patterns:

    * ``level_shard()``:  ``Shard(0)``  — shard along vertical levels.
    * ``batch_shard()``:  ``Shard(0)``  — shard along batch dimension.
    * ``replicate()``:    ``Replicate()``  — full copy on every rank.
    * ``spatial_shard(lon_or_lat)``:  ``Shard(1)`` or ``Shard(2)`` (nodal only).

    Attributes:
        placements: tuple of Placement objects (length = mesh.ndim).
    """
    placements: tuple

    # ── factory methods ────────────────────────────────────────────────

    @staticmethod
    def level_shard(mesh_ndim: int = 1) -> "DTensorShardingSpec":
        """Shard along dimension 0 (n_layers or batch).

        For a 3-D field ``(n_layers, lon, lat)`` this splits vertical
        levels across ranks on mesh dim 0 and replicates on remaining
        mesh dims (if any).
        """
        _require_dtensor()
        from torch.distributed.tensor import Shard, Replicate
        placements = [Shard(0)] + [Replicate()] * (mesh_ndim - 1)
        return DTensorShardingSpec(placements=tuple(placements))

    @staticmethod
    def batch_shard(mesh_ndim: int = 1) -> "DTensorShardingSpec":
        """Shard along the batch dimension (dim 0)."""
        return DTensorShardingSpec.level_shard(mesh_ndim)

    @staticmethod
    def replicate(mesh_ndim: int = 1) -> "DTensorShardingSpec":
        """Full replication on every rank (for basis matrices, etc.)."""
        _require_dtensor()
        from torch.distributed.tensor import Replicate
        return DTensorShardingSpec(
            placements=tuple(Replicate() for _ in range(mesh_ndim))
        )

    @staticmethod
    def spatial_shard(
        dim: int = 1,
        mesh_ndim: int = 1,
    ) -> "DTensorShardingSpec":
        """Shard along a spatial axis (nodal fields only).

        Args:
            dim: tensor dimension to shard (1 = longitude, 2 = latitude).
            mesh_ndim: number of mesh dimensions.

        .. warning::
            Sharding spatial axes requires ``redistribute`` to Replicate
            before any spectral transform (SHT).
        """
        _require_dtensor()
        from torch.distributed.tensor import Shard, Replicate
        placements = [Replicate()] * mesh_ndim
        placements[0] = Shard(dim)
        return DTensorShardingSpec(placements=tuple(placements))


def layout_to_sharding_spec(
    layout: str | None,
    mesh_ndim: int = 1,
) -> "DTensorShardingSpec":
    """Map a human-readable layout name to a DTensor sharding spec."""
    normalized = (layout or "replicate").strip().lower()
    if normalized in {"replicate", "none"}:
        return DTensorShardingSpec.replicate(mesh_ndim)
    if normalized == "level":
        return DTensorShardingSpec.level_shard(mesh_ndim)
    if normalized == "batch":
        return DTensorShardingSpec.batch_shard(mesh_ndim)
    if normalized == "spatial_lon":
        return DTensorShardingSpec.spatial_shard(dim=1, mesh_ndim=mesh_ndim)
    if normalized == "spatial_lat":
        return DTensorShardingSpec.spatial_shard(dim=2, mesh_ndim=mesh_ndim)
    raise ValueError(f"Unsupported DTensor layout: {layout}")


# ═════════════════════════════════════════════════════════════════════════
# Single tensor shard / gather
# ═════════════════════════════════════════════════════════════════════════


def shard_tensor(
    tensor: torch.Tensor,
    spec: DTensorShardingSpec,
    mesh: "torch.distributed.device_mesh.DeviceMesh",
) -> "torch.distributed.tensor.DTensor":
    """Distribute a local Tensor into a DTensor.

    Uses ``DTensor.from_local`` which is zero-copy when the local tensor
    already has the correct shard shape.

    Args:
        tensor: full (global) tensor on the current rank.
        spec: sharding specification.
        mesh: the DeviceMesh to distribute onto.

    Returns:
        A DTensor with the requested placements.
    """
    if isinstance(mesh, LocalDeviceMesh):
        return tensor
    _require_dtensor()
    from torch.distributed.tensor import distribute_tensor
    return distribute_tensor(tensor, device_mesh=mesh, placements=spec.placements)


def shard_tensor_from_local(
    local_tensor: torch.Tensor,
    spec: DTensorShardingSpec,
    mesh: "torch.distributed.device_mesh.DeviceMesh",
    global_shape: torch.Size | None = None,
) -> "torch.distributed.tensor.DTensor":
    """Create a DTensor from an already-sharded local tensor.

    Unlike ``shard_tensor`` (which scatters a full tensor), this wraps a
    pre-existing local shard without communication.

    Args:
        local_tensor: the local shard on the current rank.
        spec: sharding specification.
        mesh: DeviceMesh.
        global_shape: the global (logical) shape. If ``None``, inferred
            assuming even sharding.

    Returns:
        A DTensor backed by *local_tensor*.
    """
    if isinstance(mesh, LocalDeviceMesh):
        return local_tensor
    _require_dtensor()
    from torch.distributed.tensor import DTensor
    kwargs: dict[str, Any] = {}
    if global_shape is not None:
        kwargs["shape"] = global_shape
    return DTensor.from_local(
        local_tensor,
        device_mesh=mesh,
        placements=spec.placements,
        run_check=False,
        **kwargs,
    )


def gather_tensor(
    dtensor: "torch.distributed.tensor.DTensor",
) -> torch.Tensor:
    """Collect a DTensor back into a full Tensor (all-gather if sharded).

    Equivalent to ``dtensor.full_tensor()`` — performs the necessary
    collective and returns a plain ``torch.Tensor``.
    """
    if hasattr(dtensor, "full_tensor"):
        return dtensor.full_tensor()
    return dtensor


def redistribute_tensor(
    dtensor: "torch.distributed.tensor.DTensor",
    target_spec: DTensorShardingSpec,
    mesh: "torch.distributed.device_mesh.DeviceMesh | None" = None,
) -> "torch.distributed.tensor.DTensor":
    """Change the sharding layout of a DTensor.

    Triggers the minimal collective (all-to-all, all-gather, reduce-scatter)
    needed to move from the current placements to *target_spec*.

    Common example — before SHT on spatially-sharded nodal data::

        nodal_dt = redistribute_tensor(
            nodal_dt, DTensorShardingSpec.replicate()
        )
        modal = grid.to_modal(nodal_dt.to_local())
    """
    if not hasattr(dtensor, "redistribute"):
        return dtensor
    return dtensor.redistribute(
        device_mesh=mesh,
        placements=target_spec.placements,
    )


# ═════════════════════════════════════════════════════════════════════════
# State dict shard / gather
# ═════════════════════════════════════════════════════════════════════════


def _map_state_dict(
    fn,
    state: dict[str, Any],
) -> dict[str, Any]:
    """Apply *fn* to every Tensor leaf in a nested dict."""
    out = {}
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            out[k] = fn(v)
        elif isinstance(v, dict):
            out[k] = _map_state_dict(fn, v)
        else:
            out[k] = v
    return out


def shard_state_dict(
    state: dict[str, Any],
    mesh: "torch.distributed.device_mesh.DeviceMesh",
    field_specs: dict[str, DTensorShardingSpec] | None = None,
    default_spec: DTensorShardingSpec | None = None,
) -> dict[str, Any]:
    """Shard every Tensor in a NeuralGCM state dict.

    Args:
        state: nested dict of Tensors (e.g. ``SimulationState.__dict__``).
        mesh: DeviceMesh.
        field_specs: optional per-field overrides (key → spec).
        default_spec: fallback spec applied to fields not in *field_specs*.
            Defaults to ``level_shard(mesh.ndim)``.

    Returns:
        A new dict with DTensors in place of plain Tensors.
    """
    _require_dtensor()
    if default_spec is None:
        default_spec = DTensorShardingSpec.level_shard(mesh.ndim)
    if field_specs is None:
        field_specs = {}

    def _shard(key: str, tensor: torch.Tensor):
        spec = field_specs.get(key, default_spec)
        return shard_tensor(tensor, spec, mesh)

    out = {}
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            out[k] = _shard(k, v)
        elif isinstance(v, dict):
            # recurse (tracers, etc.)
            out[k] = {
                sub_k: _shard(f"{k}.{sub_k}", sub_v) if isinstance(sub_v, torch.Tensor) else sub_v
                for sub_k, sub_v in v.items()
            }
        else:
            out[k] = v
    return out


def gather_state_dict(
    state: dict[str, Any],
) -> dict[str, Any]:
    """Gather all DTensors in a state dict back to plain Tensors.

    Non-DTensor values are passed through unchanged.
    """
    _require_dtensor()
    from torch.distributed.tensor import DTensor

    def _gather(v):
        if isinstance(v, DTensor):
            return gather_tensor(v)
        return v

    return _map_state_dict(_gather, state)


def shard_state(
    state: State,
    mesh: "torch.distributed.device_mesh.DeviceMesh",
    field_specs: dict[str, DTensorShardingSpec] | None = None,
    default_spec: DTensorShardingSpec | None = None,
) -> State:
    """Shard a NeuralGCM State into DTensor-backed fields."""
    if default_spec is None:
        default_spec = DTensorShardingSpec.level_shard(mesh.ndim)
    if field_specs is None:
        field_specs = {}

    def _spec_for(name: str) -> DTensorShardingSpec:
        return field_specs.get(name, default_spec)

    tracers = {
        name: shard_tensor(tensor, field_specs.get(f"tracers.{name}", field_specs.get("tracers", default_spec)), mesh)
        for name, tensor in state.tracers.items()
    }
    return State(
        vorticity=shard_tensor(state.vorticity, _spec_for("vorticity"), mesh),
        divergence=shard_tensor(state.divergence, _spec_for("divergence"), mesh),
        temperature_variation=shard_tensor(
            state.temperature_variation,
            _spec_for("temperature_variation"),
            mesh,
        ),
        log_surface_pressure=shard_tensor(
            state.log_surface_pressure,
            _spec_for("log_surface_pressure"),
            mesh,
        ),
        tracers=tracers,
        sim_time=state.sim_time,
    )


def gather_state(state: State) -> State:
    """Gather a DTensor-backed NeuralGCM State back to plain tensors."""
    _require_dtensor()
    from torch.distributed.tensor import DTensor

    def _maybe_gather(tensor: torch.Tensor) -> torch.Tensor:
        if isinstance(tensor, DTensor):
            return gather_tensor(tensor)
        return tensor

    return State(
        vorticity=_maybe_gather(state.vorticity),
        divergence=_maybe_gather(state.divergence),
        temperature_variation=_maybe_gather(state.temperature_variation),
        log_surface_pressure=_maybe_gather(state.log_surface_pressure),
        tracers={k: _maybe_gather(v) for k, v in state.tracers.items()},
        sim_time=state.sim_time,
    )


def maybe_enable_dtensor_sharding(
    model: nn.Module,
    config: Any,
    coords: Any | None = None,
) -> nn.Module:
    """Create a DeviceMesh and bind state/parameter sharding helpers to a model.

    This is intentionally conservative: the mesh and state sharding helpers are
    always attached for DTensor-aware runs, while in-place parameter sharding is
    only attempted when a non-replicated parameter layout is requested.
    """
    if not getattr(config, "dtensor_aware", False):
        return model

    import torch.distributed as dist

    backend = dist.get_backend() if dist.is_initialized() else ""
    if backend == "nccl" and torch.cuda.is_available():
        model_device = "cuda"
    elif backend in {"gloo", "mpi"}:
        model_device = "cpu"
    else:
        model_device = "cuda" if any(p.is_cuda for p in model.parameters()) else "cpu"

    mesh_shape = _parse_mesh_shape(getattr(config, "dtensor_mesh_shape", ""))
    mesh_ndim = len(mesh_shape) if mesh_shape is not None else 1
    mesh_dim_names = _parse_mesh_dim_names(
        getattr(config, "dtensor_mesh_dim_names", ""),
        mesh_ndim,
    )
    mesh = create_device_mesh(
        mesh_shape=mesh_shape,
        mesh_dim_names=mesh_dim_names,
        device_type=model_device,
    )

    if coords is None:
        coords = getattr(getattr(model, "model_config", None), "coords", None)
    if coords is None:
        coords = getattr(getattr(model, "config", None), "coords", None)
    if coords is None:
        coords = getattr(model, "coords", None)

    state_default_spec = layout_to_sharding_spec(
        getattr(config, "dtensor_state_layout", "level"),
        mesh.ndim,
    )
    param_default_spec = layout_to_sharding_spec(
        getattr(config, "dtensor_param_layout", "replicate"),
        mesh.ndim,
    )

    field_specs = get_neuralgcm_field_specs(coords, mesh.ndim) if coords is not None else {}
    if getattr(config, "dtensor_state_layout", "level") != "level":
        for key in ("vorticity", "divergence", "temperature_variation", "tracers"):
            field_specs[key] = state_default_spec

    model._dtensor_mesh = mesh
    model._dtensor_runtime_mode = "local_fallback" if isinstance(mesh, LocalDeviceMesh) else "device_mesh"
    model._dtensor_field_specs = field_specs
    model._dtensor_state_default_spec = state_default_spec
    model._dtensor_param_default_spec = param_default_spec
    model._dtensor_spectral_context = SpectralShardingContext(mesh, state_default_spec)
    model.shard_state_with_dtensor = lambda state: shard_state(
        state,
        mesh,
        field_specs=field_specs,
        default_spec=state_default_spec,
    )
    model.gather_state_from_dtensor = lambda state: gather_state(state)

    param_layout = getattr(config, "dtensor_param_layout", "replicate")
    if param_layout != "replicate" and hasattr(model, "parameterization"):
        try:
            shard_module_params(
                model.parameterization,
                mesh,
                default_spec=param_default_spec,
            )
            model._dtensor_parameterization_sharded = True
            logger.info(
                "DTensor parameter sharding applied to model.parameterization with layout=%s",
                param_layout,
            )
        except Exception as exc:
            model._dtensor_parameterization_sharded = False
            logger.warning(
                "DTensor parameter sharding could not be applied (%s); continuing with bound mesh/runtime only.",
                exc,
            )

    logger.info(
        "DTensor runtime bound: mode=%s mesh=%s dims=%s state_layout=%s param_layout=%s",
        getattr(model, "_dtensor_runtime_mode", "device_mesh"),
        tuple(mesh.mesh.shape),
        getattr(mesh, "mesh_dim_names", None),
        getattr(config, "dtensor_state_layout", "level"),
        param_layout,
    )
    return model


# ═════════════════════════════════════════════════════════════════════════
# Module parameter sharding
# ═════════════════════════════════════════════════════════════════════════


def shard_module_params(
    module: nn.Module,
    mesh: "torch.distributed.device_mesh.DeviceMesh",
    param_specs: dict[str, DTensorShardingSpec] | None = None,
    default_spec: DTensorShardingSpec | None = None,
) -> None:
    """Shard nn.Module parameters in-place using DTensor.

    This converts each ``nn.Parameter`` to a DTensor-backed parameter,
    enabling transparent tensor-parallel forward/backward.

    Args:
        module: the model to shard.
        mesh: DeviceMesh.
        param_specs: per-parameter overrides ``{name: spec}``.
        default_spec: fallback (default: Replicate).

    Example::

        mesh = create_device_mesh((4,), ("tp",))
        # Shard the neural network's large linear layers column-wise
        shard_module_params(
            model.parameterization.neural_net,
            mesh,
            param_specs={"weight": DTensorShardingSpec.level_shard()},
        )
    """
    _require_dtensor()
    from torch.distributed.tensor import distribute_tensor

    if default_spec is None:
        default_spec = DTensorShardingSpec.replicate(mesh.ndim)
    if param_specs is None:
        param_specs = {}

    for name, param in list(module.named_parameters()):
        spec = param_specs.get(name, default_spec)
        dt = distribute_tensor(param.data, device_mesh=mesh, placements=spec.placements)
        # Replace the parameter with a DTensor-backed one
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, attr_name = parts
            parent = module
            for sub in parent_name.split("."):
                parent = getattr(parent, sub)
            setattr(parent, attr_name, nn.Parameter(dt, requires_grad=param.requires_grad))
        else:
            setattr(module, name, nn.Parameter(dt, requires_grad=param.requires_grad))

    logger.info(f"Sharded {sum(1 for _ in module.parameters())} parameters via DTensor")


# ═════════════════════════════════════════════════════════════════════════
# Spectral-aware sharding helpers (NeuralGCM-specific)
# ═════════════════════════════════════════════════════════════════════════


def get_neuralgcm_field_specs(
    coords: Any,
    mesh_ndim: int = 1,
) -> dict[str, DTensorShardingSpec]:
    """Return recommended DTensor specs for NeuralGCM state fields.

    Rules:
        * 3-D modal/nodal fields → ``Shard(0)`` (vertical levels)
        * surface fields (n_layers=1) → ``Replicate`` (too small to shard)
        * SHT basis matrices → ``Replicate``
        * tracers → ``Shard(0)`` (same as prognostic fields)

    Args:
        coords: ``CoordinateSystem`` with ``vertical.layers`` info.
        mesh_ndim: DeviceMesh dimensionality.

    Returns:
        dict mapping field names to sharding specs.
    """
    _require_dtensor()
    n_layers = getattr(getattr(coords, "vertical", None), "layers", None)
    level_spec = DTensorShardingSpec.level_shard(mesh_ndim)
    replicate_spec = DTensorShardingSpec.replicate(mesh_ndim)

    specs: dict[str, DTensorShardingSpec] = {
        # 3-D prognostic (modal)
        "vorticity": level_spec,
        "divergence": level_spec,
        "temperature_variation": level_spec,
        # surface (too thin to shard)
        "log_surface_pressure": replicate_spec,
        # tracers
        "tracers": level_spec,
        # SHT basis — always replicate
        "basis.f": replicate_spec,
        "basis.p": replicate_spec,
        "basis.w": replicate_spec,
    }

    # If layers ≤ mesh_size, level sharding yields empty slices → replicate
    if n_layers is not None and n_layers <= 1:
        for k in ("vorticity", "divergence", "temperature_variation"):
            specs[k] = replicate_spec

    return specs


# ═════════════════════════════════════════════════════════════════════════
# Context manager for transparent SHT with sharded tensors
# ═════════════════════════════════════════════════════════════════════════


class SpectralShardingContext:
    """Context manager that automatically redistributes around SHT calls.

    When fields are ``Shard(0)`` on levels, SHT can operate independently
    on each local level slice — no redistribution needed.

    When fields are spatially sharded ``Shard(1)`` or ``Shard(2)``, this
    context gathers to ``Replicate`` before SHT, and re-shards after.

    Usage::

        ctx = SpectralShardingContext(mesh, spec)
        # Shard nodal field
        nodal_dt = shard_tensor(nodal, spec, mesh)
        # Gather if needed, apply SHT, re-shard
        modal_dt = ctx.to_modal(grid, nodal_dt)
    """

    def __init__(
        self,
        mesh: "torch.distributed.device_mesh.DeviceMesh",
        spec: DTensorShardingSpec,
    ):
        self.mesh = mesh
        self.spec = spec
        _require_dtensor()
        from torch.distributed.tensor import Shard
        # Check if any placement shards spatial axes (dim ≥ 1 for nodal)
        self._needs_gather = any(
            isinstance(p, Shard) and p.dim >= 1
            for p in spec.placements
        )

    def to_modal(
        self,
        grid: Any,
        nodal_dt: "torch.distributed.tensor.DTensor",
    ) -> "torch.distributed.tensor.DTensor":
        """Forward SHT: nodal DTensor → modal DTensor."""
        if self._needs_gather:
            # gather spatial dims before SHT
            rep = DTensorShardingSpec.replicate(self.mesh.ndim)
            nodal_full = redistribute_tensor(nodal_dt, rep, self.mesh)
            modal_local = grid.to_modal(nodal_full.to_local())
            # re-shard result on levels
            level_spec = DTensorShardingSpec.level_shard(self.mesh.ndim)
            return shard_tensor(modal_local, level_spec, self.mesh)
        else:
            # level-sharded: SHT is per-level, no communication needed
            modal_local = grid.to_modal(nodal_dt.to_local())
            return shard_tensor_from_local(
                modal_local, self.spec, self.mesh,
            )

    def to_nodal(
        self,
        grid: Any,
        modal_dt: "torch.distributed.tensor.DTensor",
    ) -> "torch.distributed.tensor.DTensor":
        """Inverse SHT: modal DTensor → nodal DTensor."""
        nodal_local = grid.to_nodal(modal_dt.to_local())
        if self._needs_gather:
            # re-shard to the spatial spec the caller expects
            return shard_tensor(nodal_local, self.spec, self.mesh)
        else:
            return shard_tensor_from_local(
                nodal_local, self.spec, self.mesh,
            )
