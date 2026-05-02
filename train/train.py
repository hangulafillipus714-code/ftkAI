"""
train/train.py
--------------
Main entry point for training the ModernLLM.
Supports single-GPU, multi-GPU (DDP), and CPU training.

Usage:
  python train/train.py                  # Single-GPU
  torchrun --nproc_per_node=8 train/train.py  # Multi-GPU
"""

import os
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.cuda.amp import GradScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.model_config import ModelConfig
from config.train_config import TrainConfig
from model.model import ModernLLM
from data_source.tokenizer import Tokenizer
from data_source.dataset import (
    IGNORE_INDEX,
    KafkaConfig,
    QualityFilterConfig,
    create_dataloader_from_paths,
    validate_kafka_startup,
)
from optimizer.optimizer import build_optimizer
from scheduler.cosine import get_scheduled_lr, set_lr
from checkpoint.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    load_checkpoint_metadata,
    latest_checkpoint,
)
from distributed.distributed import (
    init_distributed, 
    get_device, 
    wrap_model_ddp, 
    is_main_process, 
    print_main,
    all_reduce_mean,
    barrier,
    destroy_distributed
)
from utils.seed import set_seed


def evaluate(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    device_type: str,
    amp_enabled: bool,
    amp_dtype: str,
    max_batches: int | None = None,
) -> dict[str, float]:
    was_training = model.training
    model.eval()

    total_loss = torch.zeros(1, device=device, dtype=torch.float64)
    total_tokens = torch.zeros(1, device=device, dtype=torch.float64)

    amp_context = autocast(
        device_type=device_type,
        dtype=torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16,
        enabled=amp_enabled,
    ) if amp_enabled else nullcontext()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with amp_context:
                output = model(x, attention_mask=attention_mask)
                logits = output["logits"]
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                    ignore_index=IGNORE_INDEX,
                    reduction="sum",
                )

            valid_tokens = (y != IGNORE_INDEX).sum()
            total_loss += loss.detach().to(torch.float64)
            total_tokens += valid_tokens.detach().to(torch.float64)

    total_loss = all_reduce_mean(total_loss)
    total_tokens = all_reduce_mean(total_tokens)

    denom = max(total_tokens.item(), 1.0)
    mean_loss = total_loss.item() / denom
    perplexity = float(torch.exp(torch.tensor(min(mean_loss, 20.0))).item())

    if was_training:
        model.train()

    return {
        "loss": mean_loss,
        "perplexity": perplexity,
        "tokens": total_tokens.item(),
    }


def train():
    # ── 1. Distributed Init ───────────────────────────────────────────────────
    t_cfg = TrainConfig()
    rank, local_rank, world_size = init_distributed(backend=t_cfg.backend)
    device = get_device(local_rank)
    device_type = "cuda" if device.type == "cuda" else "cpu"

    # ── 2. Config & Seed ──────────────────────────────────────────────────────
    set_seed(t_cfg.seed, deterministic=t_cfg.deterministic)
    m_cfg = ModelConfig()

    # ── 3. Tokenizer & Data ───────────────────────────────────────────────────
    tokenizer = Tokenizer(t_cfg.tokenizer_path)
    m_cfg.vocab_size = tokenizer.vocab_size  # Ensure consistency

    # If resume_from is explicitly set to an empty string or None,
    # or if no latest checkpoint is found, we start training from scratch.
    # Auto-resume logic
    ckpt_path = t_cfg.resume_from

    if ckpt_path is None:
        ckpt_path = latest_checkpoint(t_cfg.checkpoint_dir)

    if ckpt_path:
        ckpt_meta = load_checkpoint_metadata(ckpt_path, device=device)
        if ckpt_meta["model_config"]:
            m_cfg = ModelConfig.from_dict(ckpt_meta["model_config"])
            m_cfg.vocab_size = tokenizer.vocab_size

    data_paths = t_cfg.data_paths if t_cfg.data_paths else (t_cfg.data_path,)
    quality_config = QualityFilterConfig(
        min_chars=t_cfg.quality_min_chars,
        min_alpha_ratio=t_cfg.quality_min_alpha_ratio,
        max_symbol_ratio=t_cfg.quality_max_symbol_ratio,
        max_repeated_line_fraction=t_cfg.quality_max_repeated_line_fraction,
        min_unique_token_ratio=t_cfg.quality_min_unique_token_ratio,
    )
    kafka_config = KafkaConfig(
        backend=t_cfg.kafka_backend,
        bootstrap_servers=t_cfg.kafka_bootstrap_servers,
        topic=t_cfg.kafka_topic,
        group_id=t_cfg.kafka_group_id,
        auto_offset_reset=t_cfg.kafka_auto_offset_reset,
        enable_auto_commit=t_cfg.kafka_enable_auto_commit,
        poll_timeout_s=t_cfg.kafka_poll_timeout_s,
        max_empty_polls=t_cfg.kafka_max_empty_polls,
    )

    estimated_steps = t_cfg.max_steps or 100_000

    if t_cfg.use_kafka_dataset:
        print_main(
            f"[Train] Validating Kafka connection | "
            f"backend={kafka_config.backend} | "
            f"bootstrap={kafka_config.bootstrap_servers} | "
            f"topic={kafka_config.topic}"
        )
        validate_kafka_startup(kafka_config)

    dataloader = create_dataloader_from_paths(
        data_paths=data_paths,
        tokenizer=tokenizer,
        batch_size=t_cfg.batch_size,
        max_length=t_cfg.max_length,
        stride=t_cfg.stride,
        shuffle=t_cfg.shuffle,
        num_workers=t_cfg.num_workers,
        pin_memory=t_cfg.pin_memory and device_type == "cuda",
        distributed=(world_size > 1),
        rank=rank,
        world_size=world_size,
        use_curriculum=t_cfg.use_curriculum,
        total_steps=estimated_steps,
        curriculum_difficulty_window=t_cfg.curriculum_difficulty_window,
        curriculum_easy_retention=t_cfg.curriculum_easy_retention,
        drop_last=t_cfg.drop_last,
        persistent_workers=t_cfg.persistent_workers,
        use_streaming=t_cfg.use_streaming_dataset,
        use_kafka=t_cfg.use_kafka_dataset,
        kafka_config=kafka_config if t_cfg.use_kafka_dataset else None,
        use_quality_filter=t_cfg.use_quality_filter,
        quality_config=quality_config,
    )

    eval_dataloader = None
    if t_cfg.eval_data_paths:
        eval_dataloader = create_dataloader_from_paths(
            data_paths=t_cfg.eval_data_paths,
            tokenizer=tokenizer,
            batch_size=t_cfg.batch_size,
            max_length=t_cfg.max_length,
            stride=t_cfg.stride,
            shuffle=t_cfg.eval_shuffle,
            num_workers=t_cfg.num_workers,
            pin_memory=t_cfg.pin_memory and device_type == "cuda",
            distributed=(world_size > 1),
            rank=rank,
            world_size=world_size,
            use_curriculum=False,
            drop_last=False,
            persistent_workers=t_cfg.persistent_workers,
            use_streaming=t_cfg.use_streaming_dataset and not t_cfg.use_kafka_dataset,
            use_kafka=False,
            use_quality_filter=t_cfg.use_quality_filter,
            quality_config=quality_config,
        )

    # ── 4. Model ─────────────────────────────────────────────────────────────
    model = ModernLLM(
        config=m_cfg,
        gradient_checkpointing=t_cfg.gradient_checkpointing,
        mtp_heads=m_cfg.mtp_heads,
    ).to(device)
    
    model = wrap_model_ddp(
        model,
        device,
        local_rank,
        world_size,
        compile_model=t_cfg.compile_model,
        find_unused_parameters=t_cfg.find_unused_parameters,
        use_fsdp=t_cfg.use_fsdp,
    )
    raw_model = model.module if world_size > 1 else model
    
    print_main(f"[Train] Model params: {raw_model.num_parameters():,}")

    # ── 5. Optimizer & Scheduler ─────────────────────────────────────────────
    optimizer = build_optimizer(
        model=model,
        optimizer_type=t_cfg.optimizer,
        lr=t_cfg.lr,
        weight_decay=t_cfg.weight_decay,
        beta1=t_cfg.beta1,
        beta2=t_cfg.beta2,
        eps=t_cfg.eps,
        device_type=device_type,
    )
    
    # Only use GradScaler on CUDA
    amp_enabled = t_cfg.use_amp and device_type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)
    
    # ── 6. Resume Checkpoint ──────────────────────────────────────────────────
    start_step = 0
    start_epoch = 0
    
    if ckpt_path:
        ckpt_state = load_checkpoint(
            ckpt_path,
            model,
            optimizer,
            scaler,
            device,
            strict=t_cfg.strict_checkpoint_load,
        )
        start_step = ckpt_state["step"]
        start_epoch = ckpt_state["epoch"]

    # ── 7. Training Loop ─────────────────────────────────────────────────────
    try:
        dataloader_len = len(dataloader)
    except TypeError:
        dataloader_len = None

    if (t_cfg.use_streaming_dataset or t_cfg.use_kafka_dataset) and t_cfg.max_steps is None:
        raise ValueError("Streaming or Kafka training requires max_steps to be set explicitly.")

    total_steps = t_cfg.max_steps or (dataloader_len * t_cfg.num_epochs)
    global_step = start_step
    
    print_main(f"[Train] Starting at epoch {start_epoch}, step {start_step}")
    
    model.train()
    optimizer.zero_grad(set_to_none=True)
    last_loss = None
    last_epoch = start_epoch
    best_eval_loss = float("inf")

    for epoch in range(start_epoch, t_cfg.num_epochs):
        last_epoch = epoch
        if world_size > 1 and hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)
            
        for batch_idx, batch in enumerate(dataloader):
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # ── LR Schedule Update ───────────────────────────────────────────
            lr = get_scheduled_lr(
                t_cfg.scheduler,
                global_step,
                t_cfg.warmup_steps,
                total_steps,
                t_cfg.lr,
                t_cfg.min_lr,
            )
            set_lr(optimizer, lr)
            
            # ── Forward Pass ─────────────────────────────────────────────────
            # Gradient accumulation: zero grad only on update steps
            is_last_batch = (dataloader_len is not None and (batch_idx + 1) == dataloader_len)
            is_update_step = (
                (batch_idx + 1) % t_cfg.grad_accum_steps == 0
                or is_last_batch
            )
            
            amp_context = autocast(
                device_type=device_type,
                dtype=torch.bfloat16 if t_cfg.amp_dtype == "bfloat16" else torch.float16,
                enabled=amp_enabled,
            ) if amp_enabled else nullcontext()

            with amp_context:
                
                output = model(x, attention_mask=attention_mask)
                logits = output["logits"]
                aux_loss = output["aux_loss"]
                mtp_logits = output["mtp_logits"]
                
                # Main CE loss with Label Smoothing (Industry standard for 1B+ models)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    ignore_index=IGNORE_INDEX,
                    label_smoothing=0.1,
                )
                
                # MTP losses
                mtp_loss = 0.0
                if mtp_logits:
                    # Target for head k=0 is token t+2
                    # y is [b, seq], representing tokens [1:seq+1]
                    # We need tokens [2:seq+2]
                    # Since we don't have seq+2, we can only compute up to seq-1 for first head
                    for k, m_logits in enumerate(mtp_logits):
                        # Head k predicts token t + k + 2
                        # target for head k is y shifted by k+1
                        shift = k + 1
                        if shift < y.size(1):
                            m_target = y[:, shift:]
                            m_pred = m_logits[:, :-shift]
                            mtp_loss += F.cross_entropy(
                                m_pred.reshape(-1, m_pred.size(-1)),
                                m_target.reshape(-1),
                                ignore_index=IGNORE_INDEX,
                            )
                    
                    mtp_loss = mtp_loss / len(mtp_logits)
                
                total_loss = (loss + aux_loss + mtp_loss) / t_cfg.grad_accum_steps
                last_loss = loss.detach()
            
            # ── Backward Pass ────────────────────────────────────────────────
            scaler.scale(total_loss).backward()
            
            # ── Optimizer Step ───────────────────────────────────────────────
            if is_update_step:
                scaler.unscale_(optimizer)
                grad_norm = 0.0
                if t_cfg.grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg.grad_clip)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                
                # ── Logging ──────────────────────────────────────────────────
                if global_step % t_cfg.log_every == 0 and is_main_process(rank):
                    print(
                        f"Epoch {epoch} | Step {global_step} | "
                        f"Loss {loss.item():.4f} | Aux {aux_loss.item():.4f} | "
                        f"MTP {mtp_loss if isinstance(mtp_loss, float) else mtp_loss.item():.4f} | "
                        f"GradNorm {grad_norm:.2f} | "
                        f"LR {lr:.2e}"
                    )
                
                # ── Checkpointing ────────────────────────────────────────────
                if global_step % t_cfg.save_every == 0 and is_main_process(rank):
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler if amp_enabled else None,
                        step=global_step,
                        epoch=epoch,
                        loss=loss.item(),
                        model_config=m_cfg,
                        train_config=t_cfg,
                        checkpoint_dir=t_cfg.checkpoint_dir,
                        keep_last_n=t_cfg.keep_last_n,
                    )

                if (
                    eval_dataloader is not None and
                    global_step % t_cfg.eval_every == 0
                ):
                    eval_metrics = evaluate(
                        model=model,
                        dataloader=eval_dataloader,
                        device=device,
                        device_type=device_type,
                        amp_enabled=amp_enabled,
                        amp_dtype=t_cfg.amp_dtype,
                        max_batches=t_cfg.eval_max_batches,
                    )
                    if is_main_process(rank):
                        print(
                            f"[Eval] Step {global_step} | "
                            f"Loss {eval_metrics['loss']:.4f} | "
                            f"PPL {eval_metrics['perplexity']:.2f}"
                        )
                        if eval_metrics["loss"] < best_eval_loss:
                            best_eval_loss = eval_metrics["loss"]
                            save_checkpoint(
                                model=model,
                                optimizer=optimizer,
                                scaler=scaler if amp_enabled else None,
                                step=global_step,
                                epoch=epoch,
                                loss=eval_metrics["loss"],
                                model_config=m_cfg,
                                train_config=t_cfg,
                                checkpoint_dir=t_cfg.checkpoint_dir,
                                checkpoint_name="best.pt",
                                prune_old=False,
                            )
            
            if t_cfg.max_steps and global_step >= t_cfg.max_steps:
                break
        
        if t_cfg.max_steps and global_step >= t_cfg.max_steps:
            break

    # ── 8. Cleanup ───────────────────────────────────────────────────────────
    barrier()
    if is_main_process(rank) and last_loss is not None:
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler if amp_enabled else None,
            step=global_step,
            epoch=last_epoch,
            loss=last_loss.item(),
            model_config=m_cfg,
            train_config=t_cfg,
            checkpoint_dir=t_cfg.checkpoint_dir,
            keep_last_n=t_cfg.keep_last_n,
        )
    
    destroy_distributed()
    print_main("[Train] Training finished!")


if __name__ == "__main__":
    train()
