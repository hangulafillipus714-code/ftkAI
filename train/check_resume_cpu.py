"""
train/check_resume_cpu.py
-------------------------
Minimal CPU smoke test for checkpoint save/load resume.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from checkpoint.checkpoint import (
    latest_checkpoint,
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)
from config.model_config import ModelConfig
from config.train_config import TrainConfig
from model.model import ModernLLM
from optimizer.optimizer import build_optimizer
from utils.seed import set_seed

IGNORE_INDEX = -100


def encode_text(text: str, vocab_size: int) -> list[int]:
    usable_vocab = max(8, vocab_size - 1)
    return [1 + (ord(ch) % (usable_vocab - 1)) for ch in text]


def load_batches(data_path: str, max_length: int, vocab_size: int) -> list[dict[str, torch.Tensor]]:
    rows = []
    with (PROJECT_ROOT / data_path).open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            merged = f"{row['prompt']}\n{row['completion']}"
            token_ids = encode_text(merged, vocab_size=vocab_size)
            if len(token_ids) < 2:
                continue

            token_ids = token_ids[: max_length + 1]
            input_ids = token_ids[:-1]
            labels = token_ids[1:]

            pad_len = max_length - len(input_ids)
            attention_mask = [1] * len(input_ids)
            if pad_len > 0:
                input_ids += [0] * pad_len
                labels += [IGNORE_INDEX] * pad_len
                attention_mask += [0] * pad_len

            rows.append(
                {
                    "input_ids": torch.tensor([input_ids], dtype=torch.long),
                    "labels": torch.tensor([labels], dtype=torch.long),
                    "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
                }
            )

    if not rows:
        raise RuntimeError(f"No usable rows found in {data_path}")
    return rows


def build_components(train_config: TrainConfig):
    model_config = ModelConfig(vocab_size=512)
    batches = load_batches(
        data_path=train_config.data_path,
        max_length=train_config.max_length,
        vocab_size=model_config.vocab_size,
    )
    model = ModernLLM(
        config=model_config,
        gradient_checkpointing=False,
        mtp_heads=0,
    ).to(torch.device("cpu"))
    optimizer = build_optimizer(
        model=model,
        optimizer_type=train_config.optimizer,
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
        beta1=train_config.beta1,
        beta2=train_config.beta2,
        eps=train_config.eps,
        device_type="cpu",
    )
    return model_config, batches, model, optimizer


def run_one_step(model, optimizer, batch) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    x = batch["input_ids"]
    y = batch["labels"]
    attention_mask = batch["attention_mask"]

    output = model(x, attention_mask=attention_mask)
    logits = output["logits"]
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )
    total_loss = loss + output["aux_loss"]
    total_loss.backward()
    optimizer.step()

    return float(loss.item())


def main() -> None:
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / "cpu_resume_smoke"
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)

    train_config = TrainConfig(
        data_path="DATA SPOT/training_data.jsonl",
        data_paths=("DATA SPOT/training_data.jsonl",),
        eval_data_path="DATA SPOT/eval_data.jsonl",
        eval_data_paths=("DATA SPOT/eval_data.jsonl",),
        batch_size=1,
        grad_accum_steps=1,
        max_length=64,
        stride=64,
        drop_last=False,
        num_epochs=1,
        max_steps=2,
        use_amp=False,
        gradient_checkpointing=False,
        pin_memory=False,
        persistent_workers=False,
        checkpoint_dir=str(checkpoint_dir),
        save_every=1,
        keep_last_n=2,
        log_every=1,
        eval_every=1,
        eval_max_batches=1,
    )

    set_seed(train_config.seed, deterministic=False)

    model_config, batches, model, optimizer = build_components(train_config)
    first_batch = batches[0]
    first_loss = run_one_step(model, optimizer, first_batch)
    first_ckpt = save_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=None,
        step=1,
        epoch=0,
        loss=first_loss,
        model_config=model_config,
        train_config=train_config,
        checkpoint_dir=str(checkpoint_dir),
        keep_last_n=train_config.keep_last_n,
    )

    resumed_model_config, resumed_batches, resumed_model, resumed_optimizer = build_components(train_config)
    state = load_checkpoint(
        first_ckpt,
        resumed_model,
        resumed_optimizer,
        scaler=None,
        device=torch.device("cpu"),
        strict=True,
    )
    if state["step"] != 1:
        raise RuntimeError(f"Expected checkpoint step 1, got {state['step']}")

    resumed_batch = resumed_batches[1]
    resumed_loss = run_one_step(resumed_model, resumed_optimizer, resumed_batch)
    second_ckpt = save_checkpoint(
        model=resumed_model,
        optimizer=resumed_optimizer,
        scaler=None,
        step=state["step"] + 1,
        epoch=state["epoch"],
        loss=resumed_loss,
        model_config=resumed_model_config,
        train_config=train_config,
        checkpoint_dir=str(checkpoint_dir),
        keep_last_n=train_config.keep_last_n,
    )

    latest = latest_checkpoint(str(checkpoint_dir))
    metadata = load_checkpoint_metadata(second_ckpt, device=torch.device("cpu"))
    if latest != second_ckpt:
        raise RuntimeError("Latest checkpoint path does not match the resumed save.")
    if metadata["step"] != 2:
        raise RuntimeError(f"Expected resumed checkpoint step 2, got {metadata['step']}")

    print(f"[ResumeTest] first_loss={first_loss:.4f}")
    print(f"[ResumeTest] resumed_loss={resumed_loss:.4f}")
    print(f"[ResumeTest] latest_checkpoint={latest}")
    print("[ResumeTest] Checkpoint resume verified on CPU.")


if __name__ == "__main__":
    main()
