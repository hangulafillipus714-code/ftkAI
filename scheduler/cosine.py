"""
scheduler/cosine.py
-------------------
Cosine learning rate schedule with a linear warm-up phase.

Schedule shape:
                 peak_lr
                   ╱╲
                  ╱  ╲_________
                 ╱             ╲____  ← cosine decay
                ╱                    ╲
  0 ───────────╱                      ──── min_lr
       warmup    <── cosine decay ──>

Phase 1 – Linear warm-up  [0, warmup_steps):
    lr = peak_lr * (step / warmup_steps)

Phase 2 – Cosine decay   [warmup_steps, total_steps]:
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + cos(π * progress))

This is the schedule used by GPT-3, LLaMA, and most modern LLMs.
"""

import math


def get_lr(
    step: int,
    warmup_steps: int,
    total_steps: int,
    peak_lr: float,
    min_lr: float,
) -> float:
    """
    Compute the learning rate for a given training step.

    Parameters
    ----------
    step          : Current optimiser step (0-indexed).
    warmup_steps  : Number of linear warm-up steps.
    total_steps   : Total number of training steps (warmup + cosine decay).
    peak_lr       : Maximum learning rate (reached at end of warm-up).
    min_lr        : Minimum learning rate (floor of cosine decay).

    Returns
    -------
    float  :  Learning rate for this step.
    """
    # ── Guard: clamp to min_lr after training ends ────────────────────────────
    if step >= total_steps:
        return min_lr

    # ── Phase 1: Linear warm-up ───────────────────────────────────────────────
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps

    # ── Phase 2: Cosine decay ─────────────────────────────────────────────────
    decay_steps = total_steps - warmup_steps          # steps in the cosine phase
    completed   = step - warmup_steps                 # steps taken in cosine phase
    progress    = completed / decay_steps             # 0.0 → 1.0
    coeff       = 0.5 * (1.0 + math.cos(math.pi * progress))

    return min_lr + coeff * (peak_lr - min_lr)


def get_scheduled_lr(
    scheduler_type: str,
    step: int,
    warmup_steps: int,
    total_steps: int,
    peak_lr: float,
    min_lr: float,
) -> float:
    """
    Compute LR for the configured scheduler type.
    """
    scheduler = scheduler_type.lower()

    if scheduler == "cosine":
        return get_lr(step, warmup_steps, total_steps, peak_lr, min_lr)

    if scheduler == "linear":
        if step >= total_steps:
            return min_lr
        if warmup_steps > 0 and step < warmup_steps:
            return peak_lr * (step + 1) / warmup_steps
        decay_steps = max(1, total_steps - warmup_steps)
        completed = max(0, step - warmup_steps)
        progress = min(1.0, completed / decay_steps)
        return peak_lr + progress * (min_lr - peak_lr)

    if scheduler == "constant":
        if warmup_steps > 0 and step < warmup_steps:
            return peak_lr * (step + 1) / warmup_steps
        return peak_lr

    raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def set_lr(optimizer, lr: float) -> None:
    """
    Update the learning rate of every parameter group in the optimiser.
    Call this every step before optimizer.step().
    """
    for group in optimizer.param_groups:
        group["lr"] = lr
