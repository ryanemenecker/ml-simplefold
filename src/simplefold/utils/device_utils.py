#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

"""Device selection helpers and the MLX precision guard.

Importing this module (which every inference entry point does before any MLX compute) pins MLX to
exact fp32 matmuls. MLX >= 0.30 runs fp32 GEMMs on the M5 Neural Accelerators in TF32
(10-bit mantissa) by default, which moves SimpleFold outputs at the ~1e-2 level relative to the
PyTorch backend and to MLX 0.28. MLX reads MLX_ENABLE_TF32 lazily at its first matmul, so setting
it here is sufficient. Users who explicitly want the faster reduced-precision mode can export
MLX_ENABLE_TF32=1 before running; ``setdefault`` never overrides an explicit setting.
"""

import os

os.environ.setdefault("MLX_ENABLE_TF32", "0")


def get_torch_device(preference="auto"):
    """Pick the torch device for the torch backend.

    ``auto`` prefers CUDA, then Apple Metal (MPS), then CPU. Any other value is passed to
    ``torch.device`` verbatim (e.g. ``"cpu"`` to force the previous CPU-only behavior on macOS).
    """
    import torch  # local import: keep `import model.mlx` free of torch

    if preference in (None, "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)
