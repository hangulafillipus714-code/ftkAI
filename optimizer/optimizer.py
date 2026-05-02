"""
optimizer/optimizer.py
----------------------
Robust optimizer builder with correct weight decay grouping.

Design principles:
- Decay only true weight matrices (dim >= 2)
- Never decay bias or normalization parameters
- Support multiple optimizers
- CUDA fused + foreach support
- Fail loudly on invalid configs
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Literal, cast, Any


OptimizerType = Literal["adamw", "adam", "lion"]


# ==========================================================
# Parameter Grouping
# ==========================================================


def _split_decay_parameters(model: nn.Module):
    """
    Split model parameters into decay and no_decay groups.

    Rules:
    - dim >= 2 → decay
    - dim < 2  → no decay (bias, norm scale, etc.)
    """

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    if len(decay_params) == 0:
        raise RuntimeError("No decay parameters found — model may be frozen.")

    return decay_params, no_decay_params


# ==========================================================
# Optimizer Builder
# ==========================================================


def build_optimizer(
    model: nn.Module,
    optimizer_type: OptimizerType = "adamw",
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
    device_type: str = "cuda",
    use_fused: bool = True,
    use_foreach: bool = True,
) -> torch.optim.Optimizer:

    if lr <= 0:
        raise ValueError("Learning rate must be positive.")

    if weight_decay < 0:
        raise ValueError("weight_decay cannot be negative.")

    # ------------------------------------------------------
    # Split parameter groups
    # ------------------------------------------------------
    decay_params, no_decay_params = _split_decay_parameters(model)

    param_groups = []

    if weight_decay > 0:
        param_groups.append(
            {"params": decay_params, "weight_decay": weight_decay}
        )
        param_groups.append(
            {"params": no_decay_params, "weight_decay": 0.0}
        )
    else:
        # Faster path if no weight decay at all
        param_groups.append(
            {"params": decay_params + no_decay_params, "weight_decay": 0.0}
        )

    # ------------------------------------------------------
    # Logging
    # ------------------------------------------------------
    n_decay = sum(p.numel() for p in decay_params)
    n_no_decay = sum(p.numel() for p in no_decay_params)

    print(
        f"[Optimizer] decay params: {n_decay:,} | "
        f"no_decay params: {n_no_decay:,}"
    )

    # ------------------------------------------------------
    # Build optimizer
    # ------------------------------------------------------
    optimizer_type_lower = cast(str, optimizer_type).lower()

    # ---- ADAMW ------------------------------------------------
    if optimizer_type_lower == "adamw":

        kwargs: dict[str, Any] = dict(
            lr=lr,
            betas=(beta1, beta2),
            eps=eps,
        )

        if device_type == "cuda":
            # fused and foreach cannot both be True in some torch versions
            if use_fused and not use_foreach:
                kwargs["fused"] = True
            elif use_foreach and not use_fused:
                kwargs["foreach"] = True
            # if both True → prefer fused
            elif use_fused and use_foreach:
                kwargs["fused"] = True

        try:
            optimizer = torch.optim.AdamW(param_groups, **kwargs)
            print("[Optimizer] AdamW initialized.")
        except TypeError:
            # fallback if fused/foreach unsupported
            kwargs.pop("fused", None)
            kwargs.pop("foreach", None)
            optimizer = torch.optim.AdamW(param_groups, **kwargs)
            print("[Optimizer] AdamW (fallback mode).")

    # ---- ADAM -------------------------------------------------
    elif optimizer_type_lower == "adam":

        optimizer = torch.optim.Adam(  # type: ignore[assignment]
            param_groups,
            lr=lr,
            betas=(beta1, beta2),
            eps=eps,
        )
        print("[Optimizer] Adam initialized.")

    # ---- LION -------------------------------------------------
    elif optimizer_type_lower == "lion":
        try:
            from torch.optim import Lion  # type: ignore[attr-defined]
        except ImportError:
            raise RuntimeError(
                "Lion optimizer not available in this PyTorch version."
            )

        optimizer = Lion(  # type: ignore[name-defined,assignment]
            param_groups,
            lr=lr,
            betas=(beta1, beta2),
        )
        print("[Optimizer] Lion initialized.")

    else:
        raise ValueError(
            f"Unsupported optimizer type: {optimizer_type_lower}"
        )

    return optimizer