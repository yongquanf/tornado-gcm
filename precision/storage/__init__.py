# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""Mixed-precision storage package for SDA framework.

Provides:
  - SDAStorageManager: variable-level precision storage
  - ExactCodec: lossless encode/decode
  - LossyCodec: BF16/FP16 lossy encode/decode with error bounds
"""

from tornado_gcm.precision.storage.manager import SDAStorageManager
from tornado_gcm.precision.storage.codec import ExactCodec, LossyCodec

__all__ = [
    "SDAStorageManager",
    "ExactCodec",
    "LossyCodec",
]
