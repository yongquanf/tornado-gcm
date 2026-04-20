"""SDAStorageManager: variable-level mixed-precision storage.

Routes variables to ExactCodec or LossyCodec based on StorageConfig,
providing write/read/estimate_storage operations.

Usage:
    from pytorch_src.precision.storage import SDAStorageManager
    from pytorch_src.precision.sda import StorageConfig

    mgr = SDAStorageManager(StorageConfig())
    mgr.write({"vorticity": tensor_v, "tracers": tensor_t}, "output/run1")
    data = mgr.read("output/run1")
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from pytorch_src.precision.sda import StorageConfig
from pytorch_src.precision.storage.codec import ExactCodec, LossyCodec, _EncodedTensor

logger = logging.getLogger(__name__)


class SDAStorageManager:
    """Mixed-precision storage manager.

    Routes variables to exact or lossy storage based on the configured
    critical/non-critical variable lists.

    Args:
        config: StorageConfig with variable classification and compression settings.
    """

    def __init__(self, config: Optional[StorageConfig] = None) -> None:
        self._config = config or StorageConfig()
        self._exact_codec = ExactCodec(
            compressor=self._config.compressor,
            compression_level=self._config.compression_level,
        )
        self._lossy_codec = LossyCodec(
            target_dtype=self._config.lossy_dtype,
            compressor=self._config.compressor,
            compression_level=self._config.compression_level,
        )

    def _is_critical(self, name: str) -> bool:
        """Check if a variable is classified as critical (exact storage)."""
        if self._config.mode == "exact":
            return True
        if self._config.mode == "lossy":
            return False
        # 'mixed' mode: check variable lists
        if name in self._config.critical_variables:
            return True
        if name in self._config.non_critical_variables:
            return False
        # Default: treat unknown variables as critical
        return True

    def write(
        self,
        variables: dict[str, torch.Tensor],
        output_dir: str,
        step: Optional[int] = None,
    ) -> dict[str, Any]:
        """Write variables with per-variable precision routing.

        Args:
            variables: dict mapping variable name → tensor.
            output_dir: output directory path.
            step: optional time step index (appended to filenames).

        Returns:
            Summary dict with per-variable storage info.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        summary: dict[str, Any] = {}
        metadata: dict[str, dict[str, Any]] = {}

        for name, tensor in variables.items():
            is_crit = self._is_critical(name)
            suffix = f"_s{step}" if step is not None else ""
            fname = f"{name}{suffix}.bin"

            if is_crit:
                encoded = self._exact_codec.encode(tensor, self._config.exact_dtype)
                codec_name = "exact"
            else:
                encoded = self._lossy_codec.encode(tensor, verify=True)
                codec_name = "lossy"

            fpath = out_path / fname
            fpath.write_bytes(encoded.data)

            meta = {
                "codec": codec_name,
                "shape": list(encoded.shape),
                "dtype": encoded.dtype_name,
                "original_dtype": encoded.original_dtype_name,
                "compressor": encoded.compressor,
                "size_bytes": len(encoded.data),
            }
            metadata[name] = meta
            summary[name] = {
                "codec": codec_name,
                "size_bytes": len(encoded.data),
                "original_bytes": tensor.nelement() * tensor.element_size(),
            }

        # Write metadata index
        meta_path = out_path / f"_metadata{suffix if step is not None else ''}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return summary

    def read(
        self,
        input_dir: str,
        step: Optional[int] = None,
        restore_dtype: torch.dtype = torch.float32,
    ) -> dict[str, torch.Tensor]:
        """Read variables and restore to uniform dtype.

        Args:
            input_dir: directory previously written by write().
            step: time step index to load.
            restore_dtype: dtype to upcast all tensors to.

        Returns:
            Dict mapping variable name → tensor in restore_dtype.
        """
        in_path = Path(input_dir)
        suffix = f"_s{step}" if step is not None else ""
        meta_path = in_path / f"_metadata{suffix}.json"

        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        result: dict[str, torch.Tensor] = {}
        for name, meta in metadata.items():
            fname = f"{name}{suffix}.bin"
            fpath = in_path / fname
            raw_data = fpath.read_bytes()

            encoded = _EncodedTensor(
                data=raw_data,
                shape=tuple(meta["shape"]),
                dtype_name=meta["dtype"],
                original_dtype_name=meta["original_dtype"],
                compressor=meta["compressor"],
            )

            if meta["codec"] == "exact":
                tensor = self._exact_codec.decode(encoded)
            else:
                tensor = self._lossy_codec.decode(encoded, restore_dtype=restore_dtype)

            if tensor.dtype != restore_dtype:
                tensor = tensor.to(restore_dtype)

            result[name] = tensor

        return result

    def estimate_storage(
        self,
        variables: dict[str, tuple[int, ...]],
        num_steps: int = 1,
    ) -> dict[str, Any]:
        """Estimate storage requirements without writing.

        Args:
            variables: dict mapping name → shape tuple.
            num_steps: number of time steps to store.

        Returns:
            Dict with per-variable and total estimated sizes.
        """
        estimates: dict[str, dict[str, Any]] = {}
        total_exact = 0
        total_lossy = 0

        exact_bytes_per_elem = torch.tensor([], dtype=self._config.exact_dtype).element_size()
        lossy_bytes_per_elem = torch.tensor([], dtype=self._config.lossy_dtype).element_size()

        for name, shape in variables.items():
            numel = 1
            for s in shape:
                numel *= s
            is_crit = self._is_critical(name)

            bpe = exact_bytes_per_elem if is_crit else lossy_bytes_per_elem
            raw = numel * bpe * num_steps
            # Rough compression estimate: zstd ~0.7, lz4 ~0.8, none ~1.0
            ratio = {"zstd": 0.7, "lz4": 0.8}.get(self._config.compressor, 1.0)
            compressed = int(raw * ratio)

            estimates[name] = {
                "codec": "exact" if is_crit else "lossy",
                "raw_bytes": raw,
                "estimated_bytes": compressed,
            }
            if is_crit:
                total_exact += compressed
            else:
                total_lossy += compressed

        return {
            "per_variable": estimates,
            "total_exact_bytes": total_exact,
            "total_lossy_bytes": total_lossy,
            "total_bytes": total_exact + total_lossy,
            "total_mb": round((total_exact + total_lossy) / (1024 * 1024), 2),
        }
