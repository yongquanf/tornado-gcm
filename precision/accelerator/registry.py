"""AcceleratorRegistry: priority-based operator implementation registry.

Maps operator names to their best available implementation, selecting
by priority (higher = preferred) with automatic fallback.

Usage:
    registry = AcceleratorRegistry()
    registry.register("sht_legendre", triton_impl, priority=100)
    registry.register("sht_legendre", torch_impl, priority=50)
    fn = registry.get("sht_legendre")  # → triton_impl (highest priority)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class _Entry:
    """Single implementation entry in the registry."""

    name: str
    impl: Callable
    priority: int
    backend: str  # 'triton', 'torch_compile', 'eager'
    meta: dict[str, Any] = field(default_factory=dict)


class AcceleratorRegistry:
    """Priority-based operator implementation registry.

    Selects the highest-priority available implementation for each
    named operator. Supports backend filtering and fallback chains.

    Args:
        default_backend: preferred backend filter. 'auto' selects
            highest-priority regardless of backend.
    """

    def __init__(self, default_backend: str = "auto") -> None:
        self._entries: dict[str, list[_Entry]] = {}
        self._default_backend = default_backend
        self._cache: dict[str, Callable] = {}

    def register(
        self,
        name: str,
        impl: Callable,
        priority: int = 0,
        backend: str = "eager",
        **meta: Any,
    ) -> None:
        """Register an operator implementation.

        Args:
            name: operator name (e.g. "sht_legendre", "conservation_reduce").
            impl: callable implementation.
            priority: selection priority (higher wins).
            backend: backend tag ('triton', 'torch_compile', 'eager').
            **meta: additional metadata (e.g. supported dtypes).
        """
        entry = _Entry(name=name, impl=impl, priority=priority,
                       backend=backend, meta=meta)
        self._entries.setdefault(name, []).append(entry)
        # Sort descending by priority for fast lookup
        self._entries[name].sort(key=lambda e: -e.priority)
        # Invalidate cache
        self._cache.pop(name, None)
        logger.debug(
            "Registered %s [%s] priority=%d", name, backend, priority
        )

    def get(
        self, name: str, backend: Optional[str] = None
    ) -> Callable:
        """Get the best implementation for the named operator.

        Args:
            name: operator name.
            backend: override backend filter. None uses default_backend.

        Returns:
            The highest-priority implementation matching the backend.

        Raises:
            KeyError: if no implementation is registered for the name.
        """
        effective_backend = backend or self._default_backend

        cache_key = f"{name}:{effective_backend}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        entries = self._entries.get(name)
        if not entries:
            raise KeyError(
                f"No implementation registered for operator '{name}'"
            )

        if effective_backend != "auto":
            filtered = [e for e in entries if e.backend == effective_backend]
            if filtered:
                result = filtered[0].impl
                self._cache[cache_key] = result
                return result
            # Fallback to any backend
            logger.warning(
                "No '%s' impl for backend '%s', falling back to best available",
                name, effective_backend,
            )

        result = entries[0].impl
        self._cache[cache_key] = result
        return result

    def available(self, name: str) -> list[str]:
        """List available backends for an operator.

        Args:
            name: operator name.

        Returns:
            List of backend names, sorted by priority (highest first).
        """
        entries = self._entries.get(name, [])
        return [e.backend for e in entries]

    def list_operators(self) -> list[str]:
        """List all registered operator names."""
        return list(self._entries.keys())

    def info(self, name: str) -> list[dict[str, Any]]:
        """Get detailed info for all implementations of an operator."""
        entries = self._entries.get(name, [])
        return [
            {
                "backend": e.backend,
                "priority": e.priority,
                "meta": e.meta,
            }
            for e in entries
        ]

    def clear(self) -> None:
        """Remove all registered implementations."""
        self._entries.clear()
        self._cache.clear()

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)


# ═══════════════════════════════════════════════════════════════════════════
# Global registry singleton
# ═══════════════════════════════════════════════════════════════════════════

_GLOBAL_REGISTRY: Optional[AcceleratorRegistry] = None


def get_global_registry() -> AcceleratorRegistry:
    """Get (or create) the global AcceleratorRegistry singleton."""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = AcceleratorRegistry()
        _register_builtin_operators(_GLOBAL_REGISTRY)
    return _GLOBAL_REGISTRY


def _register_builtin_operators(registry: AcceleratorRegistry) -> None:
    """Register all built-in operator implementations.

    Called once when the global registry is first created.
    """
    from pytorch_src.precision.accelerator.triton_sht import (
        sht_legendre_inverse_torch,
        sht_legendre_forward_torch,
        sht_legendre_inverse_triton,
        sht_legendre_forward_triton,
        has_triton as sht_has_triton,
    )
    from pytorch_src.precision.accelerator.triton_fixers import (
        sphere_integrate_torch,
        spectral_norm_torch,
        sphere_integrate_triton,
        spectral_norm_triton,
        has_triton as fixer_has_triton,
    )

    # --- SHT Legendre (inverse) ---
    registry.register(
        "sht_legendre_inverse",
        sht_legendre_inverse_torch,
        priority=10,
        backend="eager",
        description="PyTorch einsum FP64 inverse Legendre",
    )
    if sht_has_triton():
        registry.register(
            "sht_legendre_inverse",
            sht_legendre_inverse_triton,
            priority=100,
            backend="triton",
            description="Triton FP64-acc inverse Legendre kernel",
        )

    # --- SHT Legendre (forward) ---
    registry.register(
        "sht_legendre_forward",
        sht_legendre_forward_torch,
        priority=10,
        backend="eager",
        description="PyTorch einsum FP64 forward Legendre",
    )
    if sht_has_triton():
        registry.register(
            "sht_legendre_forward",
            sht_legendre_forward_triton,
            priority=100,
            backend="triton",
            description="Triton FP64-acc forward Legendre kernel",
        )

    # --- Sphere integration ---
    registry.register(
        "sphere_integrate",
        sphere_integrate_torch,
        priority=10,
        backend="eager",
        description="PyTorch einsum FP64 sphere quadrature",
    )
    if fixer_has_triton():
        registry.register(
            "sphere_integrate",
            sphere_integrate_triton,
            priority=100,
            backend="triton",
            description="Triton FP64 2D weighted reduction",
        )

    # --- Spectral norm ---
    registry.register(
        "spectral_norm",
        spectral_norm_torch,
        priority=10,
        backend="eager",
        description="PyTorch FP64 spectral L2 norm",
    )
    if fixer_has_triton():
        registry.register(
            "spectral_norm",
            spectral_norm_triton,
            priority=100,
            backend="triton",
            description="Triton FP64 spectral L2 norm kernel",
        )

    logger.info(
        "Registered %d operators with %d total implementations",
        len(registry), sum(len(v) for v in registry._entries.values()),
    )
