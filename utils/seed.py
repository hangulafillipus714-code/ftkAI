"""
utils/seed.py
-------------
Set all relevant random seeds for reproducible training.

Covers: Python random, NumPy, PyTorch CPU, PyTorch CUDA (all GPUs),
and CUDNN determinism flags.

Note: Full determinism with CUDA requires setting CUBLAS_WORKSPACE_CONFIG
and enabling torch.use_deterministic_algorithms(True), which may reduce
performance.  By default we only set seeds (fast path).
"""

import os
import random
import torch
from typing import Optional, Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency guard
    np = None  # type: ignore


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set random seeds across all libraries for reproducibility.

    Parameters
    ----------
    seed          : Integer seed value.
    deterministic : If True, force CUDA operations to be fully deterministic
                    (may significantly slow down training on GPU).
                    Requires PyTorch ≥ 1.8.

    Notes
    -----
    Even with the same seed, DDP training may produce slightly different
    results across runs if NCCL collective operations are non-deterministic
    (this is hardware/driver dependent).
    """
    # Python built-in
    random.seed(seed)

    # NumPy
    if np is not None:
        np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA – seeds all GPUs
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)   # for multi-GPU

    if deterministic:
        # Force cuDNN to use deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

        # Required env var for some CUDA ops in deterministic mode
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        # Raise an error if any non-deterministic op is called
        torch.use_deterministic_algorithms(True)
        print(f"[Seed] Deterministic mode enabled (seed={seed})")
    else:
        # benchmark=True lets cuDNN auto-tune convolution algorithms (faster)
        torch.backends.cudnn.benchmark = True
        print(f"[Seed] Seeds set (seed={seed})")
