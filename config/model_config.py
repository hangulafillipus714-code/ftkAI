"""
config/model_config.py
----------------------
Model architecture configuration.

Defines the structural hyperparameters of the model.
Strictly validated so incompatible architectures fail early.

Designed for:
- Checkpoint reproducibility
- Architecture hashing
- Scaling experiments
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:

    # ==========================================================
    # Vocabulary
    # ==========================================================

    vocab_size: int = 64000

    # ==========================================================
    # Sequence
    # ==========================================================

    context_length: int = 128

    # ==========================================================
    # Embeddings + MLP
    # ==========================================================

    emb_dim: int = 180
    hidden_dim: int = 512
    multiple_of: int = 64  # keep FFN dims aligned while staying CPU-friendly only when testing

    # ==========================================================
    # Attention
    # ==========================================================

    n_heads: int = 6
    n_kv_heads: int = 3
    attention_bias: bool = False
    use_flash_attention: bool = False

    # ==========================================================
    # Depth
    # ==========================================================

    n_layers: int = 12

    # ==========================================================
    # Regularization
    # ==========================================================

    drop_rate: float = 0.0

    # ==========================================================
    # RoPE
    # ==========================================================

    rope_theta: float = 10_000.0
    rope_scaling_type: str = "linear"  # linear, ntk, yarn
    rope_scaling_factor: float = 1.0

    # ==========================================================
    # Stabilisation
    # ==========================================================

    use_qk_norm: bool = True
    norm_eps: float = 1e-5
    attn_scale: float | None = None  # None = 1/sqrt(head_dim)
    logit_scale: float | None = 1.0 # None = 1/sqrt(emb_dim), Fixed value if provided

    # ==========================================================
    # MoE (optional)
    # ==========================================================

    num_experts: int = 1
    num_experts_per_tok: int = 1
    moe_capacity_factor: float = 1.25
    moe_aux_loss_coef: float = 0.01

    # ==========================================================
    # MTP (optional)
    # ==========================================================

    mtp_heads: int = 0
    mtp_use_full_block: bool = False # If False, uses lightweight MTP (FFN only)

    # ==========================================================
    # Validation
    # ==========================================================

    def __post_init__(self):

        # ---------------------------
        # Basic checks
        # ---------------------------
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        if self.context_length <= 0:
            raise ValueError("context_length must be positive")

        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")

        # ---------------------------
        # Attention dimension checks
        # ---------------------------
        if self.emb_dim % self.n_heads != 0:
            raise ValueError(
                f"emb_dim ({self.emb_dim}) must be divisible by n_heads ({self.n_heads})"
            )

        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )

        if self.hidden_dim < self.emb_dim:
            raise ValueError("hidden_dim must be >= emb_dim")

        if self.hidden_dim % self.multiple_of != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be multiple_of ({self.multiple_of})"
            )

        # ---------------------------
        # Dropout safety
        # ---------------------------
        if not (0.0 <= self.drop_rate <= 1.0):
            raise ValueError("drop_rate must be between 0 and 1")

        # ---------------------------
        # RoPE safety
        # ---------------------------
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")

        if self.rope_scaling_factor <= 0:
            raise ValueError("rope_scaling_factor must be positive")
            
        if self.rope_scaling_type not in ["linear", "ntk", "yarn"]:
            raise ValueError("rope_scaling_type must be one of: linear, ntk, yarn")

        # ---------------------------
        # MoE safety
        # ---------------------------
        if self.num_experts < 1:
            raise ValueError("num_experts must be >= 1")

        if self.num_experts_per_tok < 1:
            raise ValueError("num_experts_per_tok must be >= 1")

        if self.num_experts_per_tok > self.num_experts:
            raise ValueError(
                "num_experts_per_tok cannot exceed num_experts"
            )

        if self.moe_capacity_factor <= 0:
            raise ValueError("moe_capacity_factor must be positive")

        # ---------------------------
        # Norm safety
        # ---------------------------
        if self.norm_eps <= 0:
            raise ValueError("norm_eps must be positive")

    # ==========================================================
    # Derived Properties
    # ==========================================================

    @property
    def head_dim(self) -> int:
        return self.emb_dim // self.n_heads

    @property
    def kv_dim(self) -> int:
        return self.head_dim * self.n_kv_heads

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 1

    # ==========================================================
    # Parameter Count Estimation
    # ==========================================================

    def estimate_parameters(self) -> int:
        """
        Rough parameter count estimation (decoder-only).
        """

        # Token embedding
        embed = self.vocab_size * self.emb_dim

        # Attention per layer
        attn = (
            self.emb_dim * self.emb_dim * 3  # qkv
            + self.emb_dim * self.emb_dim    # proj
        )

        # MLP per layer
        ffn = self.emb_dim * self.hidden_dim * 2

        # Total layers
        total_layers = self.n_layers * (attn + ffn)

        return embed + total_layers

    # ==========================================================
    # Serialization
    # ==========================================================

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    # ==========================================================
    # Architecture Fingerprint
    # ==========================================================

    def hash(self) -> str:
        """
        Stable architecture hash.
        Changes if architecture changes.
        """
        encoded = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.md5(encoded).hexdigest()

    # ==========================================================
    # Summary
    # ==========================================================

    def summary(self) -> str:
        params = self.estimate_parameters()
        return (
            f"Layers: {self.n_layers}\n"
            f"Embedding dim: {self.emb_dim}\n"
            f"Heads: {self.n_heads} (KV: {self.n_kv_heads})\n"
            f"Hidden dim: {self.hidden_dim}\n"
            f"Context length: {self.context_length}\n"
            f"MoE: {self.is_moe}\n"
            f"Estimated params: {params:,}\n"
            f"Architecture hash: {self.hash()}\n"
        )
