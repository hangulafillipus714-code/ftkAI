"""
config/train_config.py
----------------------
Training configuration dataclass.

Designed for:
- FTK reproducibility
- Multi-device training
- Clean checkpoint compatibility
- Hardware-aware safety
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import torch


@dataclass
class TrainConfig:

    # ==========================================================
    # Data
    # ==========================================================

    tokenizer_path: str = "tokenizer.json"
    data_path: str = "DATA SPOT/data.txt"
    data_paths: Tuple[str, ...] = ("DATA SPOT/data.txt", "DATA SPOT/data.json")
    eval_data_path: Optional[str] = "DATA SPOT/eval_data.jsonl"
    eval_data_paths: Tuple[str, ...] = ("DATA SPOT/eval_data.jsonl",)
    use_streaming_dataset: bool = False
    use_kafka_dataset: bool = False
    kafka_backend: str = "confluent-kafka"
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic: str = ""
    kafka_group_id: str = "ftkai-train"
    kafka_auto_offset_reset: str = "earliest"
    kafka_enable_auto_commit: bool = False
    kafka_poll_timeout_s: float = 1.0
    kafka_max_empty_polls: Optional[int] = None

    max_length: int = 64
    stride: int = 64
    shuffle: bool = True
    eval_shuffle: bool = False

    # ==========================================================
    # Batching
    # ==========================================================

    batch_size: int = 1
    grad_accum_steps: int = 1
    drop_last: bool = False

    # ==========================================================
    # Curriculum
    # ==========================================================

    use_curriculum: bool = False
    curriculum_difficulty_window: float = 0.20
    curriculum_easy_retention: float = 0.10
    use_quality_filter: bool = False
    quality_min_chars: int = 8
    quality_min_alpha_ratio: float = 0.20
    quality_max_symbol_ratio: float = 0.60
    quality_max_repeated_line_fraction: float = 0.50
    quality_min_unique_token_ratio: float = 0.10

    # ==========================================================
    # Training Duration
    # ==========================================================

    num_epochs: int =50
    max_steps: Optional[int] = 20000

    # ==========================================================
    # Optimizer
    # ==========================================================

    optimizer: str = "adamw"  # smaller footprint than adam or adamw
    lr: float = 1.5e-4
    min_lr: float = 1.5e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0

    # ==========================================================
    # Scheduler
    # ==========================================================

    scheduler: str = "cosine"  # cosine | linear | constant
    warmup_steps: int = 1

    # ==========================================================
    # Precision
    # ==========================================================

    use_amp: bool = False
    amp_dtype: str = "float16"  # bfloat16 | float16

    # ==========================================================
    # Memory + Loader
    # ==========================================================

    gradient_checkpointing: bool = False
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = True

    # ==========================================================
    # Distributed
    # ==========================================================

    backend: str = "nccl"  # nccl | gloo
    find_unused_parameters: bool = False

    # ==========================================================
    # Checkpointing
    # ==========================================================

    checkpoint_dir: str = "checkpoints"
    save_every: int = 1
    keep_last_n: int = 2
    resume_from: Optional[str] = None
    strict_checkpoint_load: bool = False

    # ==========================================================
    # Logging
    # ==========================================================

    log_every: int = 1
    eval_every: int = 1
    eval_max_batches: Optional[int] = 1
    compile_model: bool = False  # torch.compile (PyTorch 2+)

    # ==========================================================
    # Reproducibility
    # ==========================================================

    seed: int = 42
    deterministic: bool = False

    # ==========================================================
    # Derived Properties (auto-set)
    # ==========================================================

    device: str = "auto"

    # ==========================================================
    # Validation
    # ==========================================================

    def __post_init__(self):

        # ---------------------------
        # Data safety
        # ---------------------------
        if not self.data_paths:
            self.data_paths = (self.data_path,)
        else:
            self.data_paths = tuple(self.data_paths)

        if not self.eval_data_paths:
            self.eval_data_paths = ((self.eval_data_path,) if self.eval_data_path else ())
        else:
            self.eval_data_paths = tuple(self.eval_data_paths)

        if self.max_length < 2:
            raise ValueError("max_length must be >= 2")

        if self.stride < 1:
            raise ValueError("stride must be >= 1")

        # ---------------------------
        # Batch safety
        # ---------------------------
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        if self.grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be >= 1")

        if self.eval_max_batches is not None and self.eval_max_batches < 1:
            raise ValueError("eval_max_batches must be >= 1 when provided")

        if self.kafka_poll_timeout_s <= 0:
            raise ValueError("kafka_poll_timeout_s must be positive")

        if self.kafka_max_empty_polls is not None and self.kafka_max_empty_polls < 1:
            raise ValueError("kafka_max_empty_polls must be >= 1 when provided")

        if self.quality_min_chars < 1:
            raise ValueError("quality_min_chars must be >= 1")

        if not (0.0 <= self.quality_min_alpha_ratio <= 1.0):
            raise ValueError("quality_min_alpha_ratio must be between 0 and 1")

        if not (0.0 <= self.quality_max_symbol_ratio <= 1.0):
            raise ValueError("quality_max_symbol_ratio must be between 0 and 1")

        if not (0.0 <= self.quality_max_repeated_line_fraction <= 1.0):
            raise ValueError("quality_max_repeated_line_fraction must be between 0 and 1")

        if not (0.0 <= self.quality_min_unique_token_ratio <= 1.0):
            raise ValueError("quality_min_unique_token_ratio must be between 0 and 1")

        # ---------------------------
        # LR safety
        # ---------------------------
        if self.min_lr > self.lr:
            raise ValueError("min_lr must be <= lr")

        # ---------------------------
        # AMP safety
        # ---------------------------
        if self.amp_dtype not in ("bfloat16", "float16"):
            raise ValueError("amp_dtype must be 'bfloat16' or 'float16'")

        if not torch.cuda.is_available():
            self.use_amp = False
            self.backend = "gloo"
            self.pin_memory = False

        # ---------------------------
        # Device auto-detect
        # ---------------------------
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ---------------------------
        # Effective batch size
        # ---------------------------
        self.effective_batch_size = (
            self.batch_size * self.grad_accum_steps
        )

        # ---------------------------
        # Scheduler validation
        # ---------------------------
        if self.scheduler not in ("cosine", "linear", "constant"):
            raise ValueError("scheduler must be cosine | linear | constant")

        if self.optimizer not in ("adamw", "adam", "lion"):
            raise ValueError("optimizer must be adamw | adam | lion")

        if self.kafka_backend not in ("confluent-kafka", "kafka-python"):
            raise ValueError("kafka_backend must be confluent-kafka | kafka-python")

        if self.use_kafka_dataset and self.use_streaming_dataset:
            raise ValueError("use_kafka_dataset and use_streaming_dataset are mutually exclusive")

        if self.use_kafka_dataset:
            if not self.kafka_topic:
                raise ValueError("kafka_topic must be set when use_kafka_dataset=True")
            if not self.kafka_bootstrap_servers:
                raise ValueError("kafka_bootstrap_servers must be set when use_kafka_dataset=True")

        # ---------------------------
        # Checkpoint dir ensure
        # ---------------------------
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    # ==========================================================
    # Utilities
    # ==========================================================

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def hash(self) -> str:
        """
        Deterministic hash of training config.
        Useful for experiment tracking.
        """
        encoded = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.md5(encoded).hexdigest()

    def summary(self) -> str:
        return (
            f"Device: {self.device}\n"
            f"Batch size: {self.batch_size}\n"
            f"Grad accum: {self.grad_accum_steps}\n"
            f"Effective batch: {self.effective_batch_size}\n"
            f"LR: {self.lr} → {self.min_lr}\n"
            f"Scheduler: {self.scheduler}\n"
            f"Optimizer: {self.optimizer}\n"
            f"AMP: {self.use_amp} ({self.amp_dtype})\n"
                ) 
