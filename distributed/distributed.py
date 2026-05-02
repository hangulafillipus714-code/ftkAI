"""
distributed/distributed.py
--------------------------
Robust distributed utilities for:

- Single process
- Multi-GPU (torchrun)
- Multi-node
- CPU distributed
- AMP / torch.compile safe
- Fault-tolerant initialisation

Production hardened.
"""

from __future__ import annotations

import os
import socket
import random
from datetime import timedelta
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


# ==========================================================
# Environment Detection
# ==========================================================

def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def get_world_info() -> tuple[int, int, int]:
    """
    Returns:
        rank
        local_rank
        world_size
    """
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)
    world_size = _env_int("WORLD_SIZE", 1)
    return rank, local_rank, world_size


def is_distributed() -> bool:
    _, _, world_size = get_world_info()
    return world_size > 1


# ==========================================================
# Initialisation
# ==========================================================

def init_distributed(
    backend: str = "nccl",
    timeout_minutes: int = 30,
    verbose: bool = True,
) -> tuple[int, int, int]:

    if dist.is_available() and dist.is_initialized():
        # Prevent double initialisation
        return get_world_info()

    rank, local_rank, world_size = get_world_info()

    if world_size == 1:
        if verbose:
            print("[Distributed] Single-process mode")
        return rank, local_rank, world_size

    # Backend safety
    if backend == "nccl" and not torch.cuda.is_available():
        backend = "gloo"

    # Required for multi-node
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")

    try:
        dist.init_process_group(
            backend=backend,
            timeout=timedelta(minutes=timeout_minutes),
        )
    except Exception as e:
        raise RuntimeError(f"DDP initialisation failed: {e}")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if verbose and rank == 0:
        print(
            f"[Distributed] Initialised | "
            f"backend={backend} | "
            f"world_size={world_size} | "
            f"master={os.environ.get('MASTER_ADDR')}:{os.environ.get('MASTER_PORT')}"
        )

    return rank, local_rank, world_size


def destroy_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


# ==========================================================
# Device Utilities
# ==========================================================

def get_device(local_rank: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_type(device: torch.device) -> str:
    return device.type


# ==========================================================
# Model Wrapping
# ==========================================================

def wrap_model_ddp(
    model: torch.nn.Module,
    device: torch.device,
    local_rank: int,
    world_size: int,
    compile_model: bool = False,
    find_unused_parameters: bool = False,
) -> torch.nn.Module:

    if world_size == 1:
        if compile_model and hasattr(torch, "compile"):
            model = torch.compile(model)  # type: ignore[assignment]
        return model

    if compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]

    if device.type == "cuda":
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused_parameters,
            broadcast_buffers=False,
        )
    else:
        model = DDP(model, find_unused_parameters=find_unused_parameters)

    return model


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


# ==========================================================
# Synchronisation Helpers
# ==========================================================

def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    Average tensor across all processes.
    Useful for metrics.
    """
    if not is_distributed():
        return tensor

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor


def broadcast_object(obj, src: int = 0):
    """
    Broadcast arbitrary Python object.
    """
    if not is_distributed():
        return obj

    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


# ==========================================================
# Gradient Sync Control (for grad accumulation)
# ==========================================================

def no_sync_context(model: torch.nn.Module):
    """
    Use during gradient accumulation:
        with no_sync_context(model):
            loss.backward()
    """
    if isinstance(model, DDP):
        return model.no_sync()
    from contextlib import nullcontext
    return nullcontext()


# ==========================================================
# Rank Utilities
# ==========================================================

def is_main_process(rank: Optional[int] = None) -> bool:
    if rank is None:
        rank, _, _ = get_world_info()
    return rank == 0


def print_main(msg: str) -> None:
    if is_main_process():
        print(msg)


# ==========================================================
# Reproducibility
# ==========================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ==========================================================
# Memory Reporting
# ==========================================================

def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def log_vram(device: torch.device, prefix: str = "") -> None:
    if device.type != "cuda":
        return

    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    peak = torch.cuda.max_memory_allocated(device) / 1024**3

    tag = f"[VRAM{' ' + prefix if prefix else ''}]"
    print(
        f"{tag} alloc={allocated:.2f}GB | "
        f"reserved={reserved:.2f}GB | "
        f"peak={peak:.2f}GB"
    )
