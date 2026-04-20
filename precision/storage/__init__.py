"""Mixed-precision storage package for SDA framework.

Provides:
  - SDAStorageManager: variable-level precision storage
  - ExactCodec: lossless encode/decode
  - LossyCodec: BF16/FP16 lossy encode/decode with error bounds
"""

from pytorch_src.precision.storage.manager import SDAStorageManager
from pytorch_src.precision.storage.codec import ExactCodec, LossyCodec

__all__ = [
    "SDAStorageManager",
    "ExactCodec",
    "LossyCodec",
]
