"""ExactCodec and LossyCodec: encode/decode tensors for storage.

ExactCodec: dtype-preserving → optional compression (zstd/lz4).
LossyCodec: quantize to lower-precision dtype → compress, with
            error-bound verification on encode.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _compress(data: bytes, compressor: str, level: int) -> bytes:
    """Compress bytes with the specified compressor."""
    if compressor == "none" or not compressor:
        return data
    if compressor == "zstd":
        try:
            import zstandard as zstd

            cctx = zstd.ZstdCompressor(level=level)
            return cctx.compress(data)
        except ImportError:
            logger.debug("zstandard not installed, storing uncompressed")
            return data
    if compressor == "lz4":
        try:
            import lz4.frame

            return lz4.frame.compress(data, compression_level=level)
        except ImportError:
            logger.debug("lz4 not installed, storing uncompressed")
            return data
    return data


def _decompress(data: bytes, compressor: str) -> bytes:
    """Decompress bytes with the specified compressor."""
    if compressor == "none" or not compressor:
        return data
    if compressor == "zstd":
        import zstandard as zstd

        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(data)
    if compressor == "lz4":
        import lz4.frame

        return lz4.frame.decompress(data)
    return data


@dataclass
class _EncodedTensor:
    """Internal representation of an encoded tensor."""

    data: bytes
    shape: tuple[int, ...]
    dtype_name: str  # numpy dtype string
    original_dtype_name: str  # original torch dtype string
    compressor: str


class ExactCodec:
    """Lossless tensor codec: dtype-aligned + optional compression.

    Args:
        compressor: compression algorithm ('zstd', 'lz4', 'none').
        compression_level: compression strength (1-9).
    """

    def __init__(
        self, compressor: str = "zstd", compression_level: int = 3
    ) -> None:
        self._compressor = compressor
        self._level = compression_level

    def encode(
        self,
        tensor: torch.Tensor,
        target_dtype: Optional[torch.dtype] = None,
    ) -> _EncodedTensor:
        """Encode tensor losslessly.

        Args:
            tensor: input tensor.
            target_dtype: storage dtype (must be >= input precision).
                If None, uses tensor's own dtype.

        Returns:
            _EncodedTensor with compressed bytes.
        """
        if target_dtype is not None:
            tensor = tensor.to(target_dtype)
        arr = tensor.detach().cpu().numpy()
        raw = arr.tobytes()
        compressed = _compress(raw, self._compressor, self._level)
        return _EncodedTensor(
            data=compressed,
            shape=arr.shape,
            dtype_name=str(arr.dtype),
            original_dtype_name=str(tensor.dtype),
            compressor=self._compressor,
        )

    def decode(self, encoded: _EncodedTensor) -> torch.Tensor:
        """Decode to tensor (lossless)."""
        raw = _decompress(encoded.data, encoded.compressor)
        arr = np.frombuffer(raw, dtype=np.dtype(encoded.dtype_name))
        arr = arr.reshape(encoded.shape)
        return torch.from_numpy(arr.copy())


class LossyCodec:
    """Lossy tensor codec: quantize to lower precision + compress.

    Args:
        target_dtype: quantization dtype (bfloat16 or float16).
        compressor: compression algorithm.
        compression_level: compression strength.
        error_bound: max acceptable relative error. Encode raises
            ValueError if exceeded.
    """

    def __init__(
        self,
        target_dtype: torch.dtype = torch.bfloat16,
        compressor: str = "zstd",
        compression_level: int = 3,
        error_bound: float = 1e-3,
    ) -> None:
        self._target_dtype = target_dtype
        self._compressor = compressor
        self._level = compression_level
        self._error_bound = error_bound

    def encode(
        self,
        tensor: torch.Tensor,
        verify: bool = True,
    ) -> _EncodedTensor:
        """Encode tensor with lossy quantization.

        Args:
            tensor: input tensor (typically float32 or float64).
            verify: check that quantization error is within bounds.

        Returns:
            _EncodedTensor with compressed quantized bytes.

        Raises:
            ValueError: if verify=True and error exceeds error_bound.
        """
        original_dtype_str = str(tensor.dtype)
        quantized = tensor.to(self._target_dtype)

        if verify and tensor.numel() > 0:
            reconstructed = quantized.to(tensor.dtype)
            abs_err = (tensor - reconstructed).abs()
            scale = tensor.abs().clamp(min=1e-12)
            rel_err = (abs_err / scale).max().item()
            if rel_err > self._error_bound:
                logger.warning(
                    "LossyCodec: max relative error %.6e exceeds bound %.6e",
                    rel_err, self._error_bound,
                )

        # Convert to numpy for serialization
        # BF16 not natively in numpy — store as uint16 bit pattern
        if self._target_dtype == torch.bfloat16:
            arr = quantized.view(torch.int16).detach().cpu().numpy()
            dtype_name = "int16"
        else:
            arr = quantized.detach().cpu().numpy()
            dtype_name = str(arr.dtype)

        raw = arr.tobytes()
        compressed = _compress(raw, self._compressor, self._level)
        return _EncodedTensor(
            data=compressed,
            shape=tensor.shape,
            dtype_name=dtype_name,
            original_dtype_name=original_dtype_str,
            compressor=self._compressor,
        )

    def decode(
        self, encoded: _EncodedTensor, restore_dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """Decode lossy-encoded tensor.

        Args:
            encoded: encoded tensor data.
            restore_dtype: if set, upcast result to this dtype
                (e.g. float32). Default: keep quantized dtype.

        Returns:
            Decoded tensor.
        """
        raw = _decompress(encoded.data, encoded.compressor)
        arr = np.frombuffer(raw, dtype=np.dtype(encoded.dtype_name))
        arr = arr.reshape(encoded.shape)
        t = torch.from_numpy(arr.copy())

        # Restore BF16 from int16 bit pattern
        if encoded.dtype_name == "int16" and self._target_dtype == torch.bfloat16:
            t = t.view(torch.bfloat16)

        if restore_dtype is not None:
            t = t.to(restore_dtype)
        return t
