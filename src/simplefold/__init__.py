#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import os as _os

# Keep MLX in exact fp32 (MLX >= 0.30 defaults to TF32 matmuls on M5 GPUs); see utils/device_utils.py.
_os.environ.setdefault("MLX_ENABLE_TF32", "0")

__all__ = []
__version__ = "0.1.0"
