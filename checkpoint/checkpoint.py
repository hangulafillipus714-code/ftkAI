"""
checkpoint/checkpoint.py
------------------------
Save and load training checkpoints.

A checkpoint contains everything needed to resume training exactly:
  - model state_dict
  - optimizer state_dict
  - GradScaler state_dict (AMP)
  - current step / epoch
  - train config dict
  - model config dict

Naming convention:  checkpoints/ckpt_step_{step:07d}.pt
The manager keeps only the N most recent checkpoints to save disk space.
"""

import os
import glob
import tempfile
import torch
import torch.nn as nn

from config.model_config import ModelConfig
from config.train_config import TrainConfig


# ══════════════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    step: int,
    epoch: int,
    loss: float,
    model_config: ModelConfig,
    train_config: TrainConfig,
    checkpoint_dir: str,
    keep_last_n: int = 3,
    checkpoint_name: str | None = None,
    prune_old: bool = True,
) -> str:
    """
    Serialise training state to disk.

    Parameters
    ----------
    model          : The model (may be DDP-wrapped; we unwrap automatically).
    optimizer      : Optimiser instance.
    scaler         : AMP GradScaler (pass None if not using AMP).
    step           : Current global optimiser step.
    epoch          : Current epoch index.
    loss           : Most recent training loss (for display / comparison).
    model_config   : ModelConfig dataclass.
    train_config   : TrainConfig dataclass.
    checkpoint_dir : Directory to write checkpoints.
    keep_last_n    : Retain only the N most recent checkpoints.

    Returns
    -------
    str  :  Path of the saved checkpoint file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Handle FSDP state dict correctly
    if hasattr(model, "module") and type(model).__name__ == "FullyShardedDataParallel":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            model_state = model.state_dict()
            opt_state = FSDP.full_optim_state_dict(model, optimizer)
    else:
        raw_model = model.module if hasattr(model, "module") else model
        model_state = raw_model.state_dict()
        opt_state = optimizer.state_dict()

    state = {
        "step":         step,
        "epoch":        epoch,
        "loss":         loss,
        "model":        model_state,
        "optimizer":    opt_state,
        "scaler":       scaler.state_dict() if scaler is not None else None,
        "model_config": model_config.to_dict(),
        # train_config stored as plain dict for portability
        "train_config": {
            k: getattr(train_config, k)
            for k in train_config.__dataclass_fields__
        },
    }

    path = os.path.join(
        checkpoint_dir,
        checkpoint_name or f"ckpt_step_{step:07d}.pt",
    )
    fd, tmp_path = tempfile.mkstemp(
        prefix=".tmp_ckpt_",
        suffix=".pt",
        dir=checkpoint_dir,
    )
    os.close(fd)

    try:
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    print(f"[Checkpoint] Saved → {path}  (step={step}, loss={loss:.4f})")

    # ── Prune old checkpoints ──────────────────────────────────────────────────
    if prune_old:
        _prune_checkpoints(checkpoint_dir, keep_last_n)

    return path


# ══════════════════════════════════════════════════════════════════════════════
# Load
# ══════════════════════════════════════════════════════════════════════════════

def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    device: torch.device = torch.device("cpu"),
    strict: bool = True,
) -> dict:
    """
    Load a checkpoint into an existing model (and optionally optimizer/scaler).

    Parameters
    ----------
    path       : Path to the .pt checkpoint file.
    model      : Model instance to load weights into.
    optimizer  : If provided, restore optimizer state too.
    scaler     : If provided, restore GradScaler state too.
    device     : Map tensors to this device when loading.

    Returns
    -------
    dict containing: step, epoch, loss, model_config, train_config
    """
    print(f"[Checkpoint] Loading ← {path}")
    state = torch.load(path, map_location=device)

    # Unwrap DDP if necessary
    raw_model = model.module if hasattr(model, "module") else model
    missing_keys, unexpected_keys = raw_model.load_state_dict(state["model"], strict=strict)
    if not strict and (missing_keys or unexpected_keys):
        print(
            "[Checkpoint] Non-strict load "
            f"(missing={len(missing_keys)}, unexpected={len(unexpected_keys)})"
        )

    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
        print("[Checkpoint] Restored optimizer state")

    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
        print("[Checkpoint] Restored GradScaler state")

    print(
        f"[Checkpoint] Resumed from step={state['step']}, "
        f"epoch={state['epoch']}, loss={state['loss']:.4f}"
    )

    return {
        "step":         state["step"],
        "epoch":        state["epoch"],
        "loss":         state["loss"],
        "model_config": state.get("model_config", {}),
        "train_config": state.get("train_config", {}),
    }


def load_checkpoint_metadata(
    path: str,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Load checkpoint metadata without applying weights to a model.
    """
    state = torch.load(path, map_location=device)
    return {
        "step": state.get("step", 0),
        "epoch": state.get("epoch", 0),
        "loss": state.get("loss"),
        "model_config": state.get("model_config", {}),
        "train_config": state.get("train_config", {}),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def latest_checkpoint(checkpoint_dir: str) -> str | None:
    """
    Find the most recent checkpoint file in `checkpoint_dir`.
    Returns None if the directory is empty or does not exist.
    """
    pattern = os.path.join(checkpoint_dir, "ckpt_step_*.pt")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def _prune_checkpoints(checkpoint_dir: str, keep_last_n: int) -> None:
    """Delete all but the `keep_last_n` most recent checkpoints."""
    pattern = os.path.join(checkpoint_dir, "ckpt_step_*.pt")
    files = sorted(glob.glob(pattern))              # oldest → newest
    to_delete = files[: max(0, len(files) - keep_last_n)]
    for f in to_delete:
        os.remove(f)
        print(f"[Checkpoint] Pruned old checkpoint: {f}")
