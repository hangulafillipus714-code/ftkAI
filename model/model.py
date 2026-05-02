"""
model/model.py
--------------
Author  : Fillipus Hangula
Date    : 2026-04-09
License : MIT

Modern LLaMA-style decoder transformer.

Components
──────────
  RMSNorm              – Pre-norm, no mean subtraction (cheaper than LayerNorm)
  RoPE                 – Rotary positional embeddings applied to Q and K only
  SwiGLU               – Gated feed-forward (SiLU gate × linear), no bias
  GroupedQueryAttention– GQA: n_kv_heads ≤ n_heads, KV shared across head groups
  TransformerBlock     – Pre-norm residual block with DeepNet residual scaling
  ModernLLM            – Full model with weight-tied input/output embeddings

fixes applied
────────────────────────
1  GroupedQueryAttention – attn_scale default was 1.0, passing scale=1.0
           to F.scaled_dot_product_attention bypasses the standard 1/√d_head
           normalisation entirely, inflating pre-softmax logits by up to 11×
           and causing gradient explosion. Default is now None so PyTorch uses
           its own 1/√d_head; explicit overrides (e.g. for QK-Norm regimes)
           are still supported via config.attn_scale.

2  SparseMoE – capacity_per_expert was computed as
           ceil(capacity_factor * T / num_experts), which is the Switch-
           Transformer formula for top-1 routing. With top_k > 1 the average
           load per expert is T * top_k / num_experts, so the old formula
           dropped ≈ 37 % of token-expert assignments at top_k=2. Corrected
           to ceil(capacity_factor * T * top_k / num_experts).

3  SparseMoE – removed eight lines of dead code (flat_expert_ids,
           flat_weights, token_ids, sort_idx, sorted_expert, sorted_token,
           sorted_weights) that were computed after the router block but
           overwritten in full by the vectorised dispatch section and never
           consumed anywhere.
4  precompute_freqs_cis (YaRN branch) – the original implementation
           applied both position-domain scaling (t /= long_factor) AND
           frequency-domain scaling (inv_freq *= long_factor^(freq/d)), which
           double-scales low-frequency components and inconsistently
           compounds high-frequency ones. Replaced with clean NTK-aware theta
           rescaling (same mathematical family as the NTK branch; YaRN's
           NTK-by-parts per-dimension blending requires beta_fast/beta_slow
           config params not present here). Removed the dead inline comment
           from the NTK branch as well.

5  ModernLLM.forward – both branches of
           isinstance(self.logit_scale, nn.Parameter) produced byte-for-byte
           identical code. Collapsed to a single unconditional assignment.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as grad_checkpoint

from config.model_config import ModelConfig
from kv_cache.cache import KVCache


# ══════════════════════════════════════════════════════════════════════════════
# Normalisation
# ══════════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalisation.

    Cheaper than LayerNorm because it skips the mean-centering step.
    Formula:  x / RMS(x)  *  γ     where  RMS(x) = sqrt(mean(x²) + ε)

    "Root Mean Square Layer Normalization"
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))   # learnable scale γ

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FP16 fix: x.pow(2) can overflow/underflow in float16 before
        # the mean is computed.  Cast to float32 for the normalisation step,
        # then cast back to the original dtype (preserves AMP compatibility).
        x_fp32   = x.float()
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(variance + self.eps)
        return (x_normed * self.weight).type_as(x)


# ══════════════════════════════════════════════════════════════════════════════
# Rotary Positional Embeddings (RoPE)
# ══════════════════════════════════════════════════════════════════════════════

def precompute_freqs_cis(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10_000.0,          # modern default; 10k
    rope_scaling_type: str = "linear", # linear | ntk | yarn
    rope_scaling_factor: float = 1.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Precompute complex-exponential rotation frequencies for RoPE.
    Supports scaling for extended context windows (1 M+).

    Returns
    -------
    Tensor of shape [max_seq_len, head_dim // 2] (complex64)

    Each position t gets frequencies:  exp(i · t · θ_j)

    Safe for FP16/AMP.  Device-aware.  Enforces even head_dim.

    Scaling modes
    ─────────────
    linear : positions are divided by rope_scaling_factor (PI / linear interp).
    ntk    : theta is rescaled so high-frequency components are preserved while
             low-frequency components are interpolated — "NTK-aware" RoPE.
             Formula: θ_new = θ · s^(d/(d-2)) where s = rope_scaling_factor.
    yarn   : NTK-aware theta rescaling identical to the ntk branch.
             Full YaRN (NTK-by-parts per-dimension blending with beta_fast /
             beta_slow) requires additional config parameters not present in
             this signature; use the ntk mode for a mathematically equivalent
             aggregate result, or extend this function with those parameters.

    4: The original yarn branch applied both t /= long_factor  AND
    inv_freq *= long_factor^(freq/d), compounding the two scaling axes and
    creating inconsistent behaviour across frequency groups.  It now uses the
    same clean NTK theta-rescaling as the ntk branch.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")

    if head_dim <= 2:
        raise ValueError(
            f"head_dim must be > 2 for NTK/YaRN RoPE scaling (got {head_dim}). "
            f"The formula head_dim/(head_dim-2) requires head_dim > 2."
        )

    # Compute inverse frequencies in float32 (stable)
    freq_seq = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (theta ** (freq_seq / head_dim))

    # Position indices
    t = torch.arange(max_seq_len, dtype=torch.float32, device=device)

    # Apply RoPE scaling
    if rope_scaling_type == "linear":
        # Position interpolation: divide positions by the scale factor.
        # Equivalent to dividing all frequencies by rope_scaling_factor.
        t = t / rope_scaling_factor

    elif rope_scaling_type == "ntk":
        # NTK-aware RoPE: rescale theta so that high-frequency components
        # (short wavelengths) are extrapolated while low-frequency components
        # are smoothly interpolated.
        # Derivation: θ_new = θ · s^(d/(d-2))  where s = rope_scaling_factor.
        ntk_theta = theta * (rope_scaling_factor ** (head_dim / (head_dim - 2)))
        inv_freq  = 1.0 / (ntk_theta ** (freq_seq / head_dim))
        # Positions are NOT scaled; the frequency axis absorbs the extension.

    elif rope_scaling_type == "yarn":
        # 4: Simplified YaRN — NTK-aware theta rescaling.
        # The original code applied BOTH position scaling (t /= s) AND
        # frequency scaling (inv_freq *= s^(freq/d)), which double-scales
        # low-frequency components and inconsistently affects high-frequency
        # ones.  The corrected form applies only theta rescaling, matching
        # the NTK aggregate behaviour that YaRN is built upon.
        #
        # Full YaRN (per-dimension linear/NTK blending via beta_fast and
        # beta_slow, plus the mscale attention correction factor) requires
        # those hyperparameters to be surfaced in this function's signature;
        # extend accordingly if needed.
        yarn_theta = theta * (rope_scaling_factor ** (head_dim / (head_dim - 2)))
        inv_freq   = 1.0 / (yarn_theta ** (freq_seq / head_dim))
        # Positions are NOT scaled; frequency axis absorbs the extension.

    else:
        raise ValueError(f"Unsupported RoPE scaling type: {rope_scaling_type}")

    # Outer product: [max_seq_len, head_dim/2]
    freqs = torch.outer(t, inv_freq)

    # Convert to complex exponential
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64

    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,        # [batch, seq_len, n_heads,    head_dim]
    xk: torch.Tensor,        # [batch, seq_len, n_kv_heads, head_dim]
    freqs_cis: torch.Tensor, # [seq_len, head_dim // 2]   (complex)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE to query and key tensors.

    Pairs each consecutive (x_{2i}, x_{2i+1}) into a complex number,
    rotates it by the precomputed frequency, then unpacks back to real.
    """
    # Reshape to complex:  last dim pairs → complex dimension
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Broadcast freqs_cis over batch and heads:  [1, seq, 1, head_dim/2]
    freqs = freqs_cis.view(1, xq_.shape[1], 1, freqs_cis.shape[-1])

    # Rotate and unpack back to real
    xq_out = torch.view_as_real(xq_ * freqs).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs).flatten(3)

    # Correctness invariant: freqs_cis must be sliced by the CALLER with
    # the correct positional offset (cache_len : cache_len + seq_len).
    # If cache_len drifts out of sync with actual cached tokens, RoPE
    # rotations will be applied at the wrong positions silently.
    # This is enforced by kv_cache.seq_len in ModernLLM.forward.
    return xq_out.type_as(xq), xk_out.type_as(xk)


def build_attention_mask(
    attention_mask: torch.Tensor,
    query_len: int,
    key_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Build an additive attention mask for SDPA from a 2D/4D padding mask.

    Input conventions:
      * [batch, key_len] with 1/True for valid tokens and 0/False for padding
      * [batch, 1, query_len, key_len] additive mask already aligned for SDPA
    """
    if attention_mask.dim() == 4:
        return attention_mask.to(device=device, dtype=dtype)

    if attention_mask.dim() != 2:
        raise ValueError(
            f"attention_mask must be 2D or 4D, got shape {tuple(attention_mask.shape)}"
        )

    if attention_mask.size(1) != key_len:
        raise ValueError(
            f"attention_mask key length ({attention_mask.size(1)}) does not match "
            f"expected key_len ({key_len})"
        )

    valid_keys = attention_mask.to(device=device, dtype=torch.bool)
    causal = torch.ones(query_len, key_len, device=device, dtype=torch.bool).tril(
        diagonal=key_len - query_len
    )
    allowed = causal.unsqueeze(0) & valid_keys.unsqueeze(1)

    additive_mask = torch.zeros(
        allowed.shape,
        device=device,
        dtype=torch.float32,
    )
    additive_mask.masked_fill_(~allowed, float("-inf"))
    return additive_mask.unsqueeze(1).to(dtype=dtype)


# ══════════════════════════════════════════════════════════════════════════════
# Feed-Forward: SwiGLU
# ══════════════════════════════════════════════════════════════════════════════

class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward network.

    output = W2 · (SiLU(W1 · x)  ⊗  W3 · x)

    The gate (W3) allows the network to selectively suppress activations.
    No bias is used (following LLaMA convention).

    "GLU Variants Improve Transformer"
    """

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)   # gate pre-activation
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)   # output projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)   # value stream

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class SparseMoE(nn.Module):
    """
    Sparse Mixture of Experts with batched scatter routing.

    Improvements over the naive loop version
    ─────────────────────────────────────────
    * Batched dispatch : tokens are grouped per expert and forwarded in a
      single batched call — no serial Python loop, full GPU utilisation.
    * Load-balancing loss: auxiliary loss (weighted ~0.01) penalises routing
      collapse. Without it, 1-2 experts absorb all tokens within 1 k steps.
      Formula: L_aux = num_experts · Σ(f_i · p_i)
      where f_i = fraction of tokens routed to expert i  (stop-gradient),
            p_i = mean routing probability for expert i  (differentiable).

    Returns
    -------
    (output, aux_loss) — caller MUST add aux_loss * aux_weight to the
    total loss during training. Pass aux_weight=0.0 at inference.

    "Switch Transformers"
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_experts     = getattr(config, "num_experts", 8)
        self.top_k           = getattr(config, "num_experts_per_tok", 2)
        self.capacity_factor = getattr(config, "moe_capacity_factor", 1.25)
        self.gate            = nn.Linear(config.emb_dim, self.num_experts, bias=False)
        self.experts         = nn.ModuleList(
            [SwiGLU(config.emb_dim, config.hidden_dim) for _ in range(self.num_experts)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, seq_len, d = x.shape
        x_flat = x.view(-1, d)   # [T, d]  where T = b * seq_len
        T = x_flat.size(0)

        # ── 1. Router ─────────────────────────────────────────────────────────
        router_logits   = self.gate(x_flat)                              # [T, E]
        router_probs    = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(router_probs, self.top_k, dim=-1)

        # Normalise selected weights so they sum to 1 per token
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(x.dtype)                          # [T, K]

        # ── 2. Load-balancing auxiliary loss ──────────────────────────────────
        # f_i : fraction of tokens assigned to expert i  (stop-gradient)
        # p_i : mean router probability for expert i     (differentiable)
        with torch.no_grad():
            # Use only the top-1 choice for load-balancing statistics
            expert_mask_hard = torch.zeros(T, self.num_experts, device=x.device)
            expert_mask_hard.scatter_(1, topk_ids, 1.0)  # All top-k choices
            f_i = expert_mask_hard.mean(0)
        p_i      = router_probs.mean(0)
        aux_loss = self.num_experts * (f_i * p_i).sum()                  # scalar

        # ── 3. Vectorised expert dispatch ─────────────────────────────────────
        # 3: Removed eight lines of dead code that constructed
        # flat_expert_ids / flat_weights / token_ids / sorted_* variables
        # which were immediately overwritten by this vectorised path and
        # never referenced anywhere in the actual computation.

        output = torch.zeros_like(x_flat)

        # 2: capacity_per_expert must account for top_k.
        # Average assignments per expert = T * top_k / num_experts.
        # The original formula (T / num_experts) is correct only for top-1
        # routing; with top_k=2 it was ~2× too small and dropped ≈37 % of
        # token-expert assignments even under balanced routing.
        capacity_per_expert = int(
            math.ceil(self.capacity_factor * T * self.top_k / self.num_experts)
        )

        # Flatten [T, K] → [T*K] keeping token↔expert↔weight aligned
        flat_tok = torch.arange(T, device=x.device).unsqueeze(1).expand(T, self.top_k).reshape(-1)  # [T*K]
        flat_exp = topk_ids.reshape(-1)         # [T*K]
        flat_w   = topk_weights.reshape(-1)     # [T*K]

        # Sort by (expert, token) so cumsum gives a stable per-expert rank
        sort_idx = (flat_exp * T + flat_tok).argsort()
        flat_tok = flat_tok[sort_idx]
        flat_exp = flat_exp[sort_idx]
        flat_w   = flat_w[sort_idx]

        # Compute each assignment's rank within its expert via cumsum.
        # expert_one_hot: [T*K, E] — 1 at the assigned expert column.
        expert_one_hot = torch.zeros(
            T * self.top_k, self.num_experts,
            device=x.device, dtype=torch.float32,
        )
        expert_one_hot.scatter_(1, flat_exp.unsqueeze(1), 1)
        # cumsum along the assignment axis gives a running count per expert
        rank = expert_one_hot.cumsum(0)[
            torch.arange(T * self.top_k, device=x.device), flat_exp
        ]  # [T*K] — 1-based rank of this assignment within its expert

        # Drop assignments that exceed expert capacity
        keep     = rank <= capacity_per_expert          # [T*K] bool mask
        flat_tok = flat_tok[keep]
        flat_exp = flat_exp[keep]
        flat_w   = flat_w[keep]

        if flat_tok.numel() == 0:
            return output.view(b, seq_len, d), aux_loss

        flat_inp = x_flat[flat_tok]   # [N_kept, d] — gather token vectors

        # Sort kept assignments by expert for contiguous batched execution
        sort_by_expert = flat_exp.argsort()
        flat_tok       = flat_tok[sort_by_expert]
        flat_exp       = flat_exp[sort_by_expert]
        flat_w         = flat_w[sort_by_expert]
        flat_inp       = flat_inp[sort_by_expert]

        # Execute each expert on its contiguous token batch
        unique_experts, counts = flat_exp.unique_consecutive(return_counts=True)
        input_batches = torch.split(flat_inp, counts.tolist(), dim=0)

        expert_outputs = []
        for i, eid in enumerate(unique_experts):
            expert_outputs.append(self.experts[eid.item()](input_batches[i]))

        all_expert_outputs = torch.cat(expert_outputs, dim=0)  # [N_kept, d]
        output.scatter_add_(
            0,
            flat_tok.unsqueeze(-1).expand_as(all_expert_outputs),
            all_expert_outputs * flat_w.unsqueeze(-1),
        )

        return output.view(b, seq_len, d), aux_loss


# ══════════════════════════════════════════════════════════════════════════════
# Grouped Query Attention (GQA)
# ══════════════════════════════════════════════════════════════════════════════

class GroupedQueryAttention(nn.Module):
    """
    Multi-head attention with Grouped Query Attention (GQA).

    n_kv_heads ≤ n_heads:
      - n_kv_heads == n_heads  →  standard MHA
      - n_kv_heads == 1        →  Multi-Query Attention (MQA)
      - 1 < n_kv_heads < n_heads → GQA (LLaMA-2/3 style)

    KV heads are stored in the cache at n_kv_heads resolution and expanded
    (repeat_interleave) *after* the cache update to save cache memory.

    "GQA: Training Generalised Multi-Query
               Transformer Models from Multi-Head Checkpoints"
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads    = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep      = self.n_heads // self.n_kv_heads   # expansion factor
        self.head_dim   = config.head_dim

        # Projection matrices (no bias – LLaMA convention)
        self.wq = nn.Linear(config.emb_dim, self.n_heads    * self.head_dim, bias=False)
        self.wk = nn.Linear(config.emb_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.emb_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads    * self.head_dim, config.emb_dim, bias=False)

        # QK-Norm: Crucial for stabilising attention entropy at 1 M+ context lengths
        self.use_qk_norm = getattr(config, "use_qk_norm", False)
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)   # per-head, not total
            self.k_norm = RMSNorm(self.head_dim)

        self.attn_drop  = nn.Dropout(config.drop_rate)
        self.resid_drop = nn.Dropout(config.drop_rate)

        # FIX-1: attn_scale default was 1.0, which bypasses the standard
        # 1/√d_head normalisation in F.scaled_dot_product_attention.
        # With head_dim=128 this inflated pre-softmax logits by 11.3×,
        # causing exploding gradients from the very first training step.
        #
        # Correct default is None: PyTorch then applies 1/√d_head internally.
        # Set config.attn_scale to a float to override (e.g. when QK-Norm is
        # active and you want scale=1.0 or a custom value).
        self.attn_scale = getattr(config, "attn_scale", None)  # None → 1/√head_dim

    def forward(
        self,
        x: torch.Tensor,           # [batch, seq_len, emb_dim]
        freqs_cis: torch.Tensor,   # [seq_len, head_dim/2]  complex RoPE freqs
        mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        layer_idx: int | None = None,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:

        b, seq_len, _ = x.shape

        # ── Project to Q, K, V ───────────────────────────────────────────────
        q_proj = self.wq(x)
        k_proj = self.wk(x)
        v = self.wv(x).view(b, seq_len, self.n_kv_heads, self.head_dim)

        # Reshape FIRST, then apply per-head norm, then RoPE
        q = q_proj.view(b, seq_len, self.n_heads,    self.head_dim)
        k = k_proj.view(b, seq_len, self.n_kv_heads, self.head_dim)

        if self.use_qk_norm:
            q = self.q_norm(q)   # applied per head: [..., head_dim]
            k = self.k_norm(k)

        # ── Apply RoPE to Q and K ─────────────────────────────────────────────
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # ── Transpose to [batch, heads, seq, head_dim] for SDPA ──────────────
        q = q.transpose(1, 2).contiguous()   # [b, n_heads,    seq, head_dim]
        k = k.transpose(1, 2).contiguous()   # [b, n_kv_heads, seq, head_dim]
        v = v.transpose(1, 2).contiguous()   # [b, n_kv_heads, seq, head_dim]

        # ── KV Cache update (at n_kv_heads resolution) ───────────────────────
        if kv_cache is not None:
            if layer_idx is None:
                raise ValueError("layer_idx must be provided when using kv_cache")
            kv_cache.update(layer_idx, k, v)
            
            from model.triton_attention import paged_attention
            
            # The attention kernel reads directly from the physical paged blocks
            out = paged_attention(
                q=q,
                physical_cache=kv_cache.physical_cache,
                block_tables=kv_cache.block_tables,
                seq_lengths=kv_cache.seq_lengths,
                layer_idx=layer_idx,
                scale=self.attn_scale,
                attn_mask=mask,
            )
            
            out = out.transpose(1, 2).contiguous().view(b, seq_len, -1)
            return self.resid_drop(self.wo(out))

        # ── Expand KV for GQA (Standard Training Path) ────────────
        if self.n_rep > 1:
            # Zero-allocation view expansion (prevents OOM crashes)
            seq_l = k.size(2)
            k = k.unsqueeze(2).expand(b, self.n_kv_heads, self.n_rep, seq_l, self.head_dim).reshape(b, self.n_heads, seq_l, self.head_dim)
            v = v.unsqueeze(2).expand(b, self.n_kv_heads, self.n_rep, seq_l, self.head_dim).reshape(b, self.n_heads, seq_l, self.head_dim)

        # ── Scaled Dot-Product Attention ──────────────────────────────────────
        # PyTorch's SDPA uses Flash Attention under the hood when available.
        #
        # Causal masking logic:
        # We use is_causal=True only if no custom mask is provided and we have
        # a sequence (seq_len > 1). During standard autoregressive decoding
        # (seq_len == 1), causality is implicit.
        #
        # This prevents the "is_causal and attn_mask are mutually exclusive"
        # error.
        attn_mask = mask
        if attention_mask is not None:
            key_len = k.size(2)
            attn_mask = build_attention_mask(
                attention_mask=attention_mask,
                query_len=seq_len,
                key_len=key_len,
                dtype=q.dtype,
                device=q.device,
            )

        is_causal = (attn_mask is None) and (seq_len > 1)

        # FIX-1: scale=self.attn_scale (None by default).
        # When None, PyTorch applies its standard 1/√d_head.
        # When a float is provided via config.attn_scale, that exact value
        # is used as the full scale factor (not a multiplier).
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            scale=self.attn_scale,   # None → default 1/√head_dim
        )

        # ── Merge heads and project out ───────────────────────────────────────
        out = out.transpose(1, 2).contiguous().view(b, seq_len, -1)
        return self.resid_drop(self.wo(out))


# ══════════════════════════════════════════════════════════════════════════════
# Transformer Block
# ══════════════════════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    """
    Pre-norm transformer block with DeepNet residual scaling.

    Layout (Pre-LN):
        x → RMSNorm → Attention  → x  (with scale)
        x → RMSNorm → SwiGLU FFN → x  (with scale)

    Residual scale = (2 * n_layers)^(-0.5) keeps gradients stable at depth
    without requiring a warm-up schedule for the residual magnitudes.

    "DeepNet: Scaling Transformers to 1,000 Layers"
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention = GroupedQueryAttention(config)

        # Config-driven MoE toggle
        self.use_moe = getattr(config, "num_experts", 1) > 1
        self.feed_forward: SparseMoE | SwiGLU
        if self.use_moe:
            self.feed_forward = SparseMoE(config)
        else:
            self.feed_forward = SwiGLU(config.emb_dim, config.hidden_dim)

        self.attention_norm = RMSNorm(config.emb_dim)
        self.ffn_norm       = RMSNorm(config.emb_dim)
        self.res_scale      = (2 * config.n_layers) ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        layer_idx: int | None = None,
        kv_cache: KVCache | None = None,
        mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 1. Attention + Residual
        normed_x = self.attention_norm(x)
        attn_out = self.attention(
            normed_x,
            freqs_cis,
            mask=mask,
            attention_mask=attention_mask,
            layer_idx=layer_idx,
            kv_cache=kv_cache,
        )
        x = x + self.res_scale * attn_out

        # 2. Feed-Forward + Residual
        normed_x = self.ffn_norm(x)
        if self.use_moe:
            ffn_out, aux_loss = self.feed_forward(normed_x)
        else:
            ffn_out  = self.feed_forward(normed_x)
            aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        x = x + self.res_scale * ffn_out
        return x, aux_loss


# ══════════════════════════════════════════════════════════════════════════════
# Multi-Token Prediction (MTP)
# ══════════════════════════════════════════════════════════════════════════════

def _slice_attention_mask(mask: torch.Tensor, new_len: int) -> torch.Tensor:
    """Safely slice attention mask for MTP head."""
    if mask.dim() == 2:
        return mask[:new_len, :new_len]
    elif mask.dim() == 3:
        return mask[:, :new_len, :new_len]
    elif mask.dim() == 4:
        return mask[:, :, :new_len, :new_len]
    else:
        raise ValueError(f"Unexpected mask dimensionality: {mask.dim()}")

class MultiTokenPredictionHead(nn.Module):
    """
    Multi-Token Prediction (MTP) heads.

    During training, K lightweight heads each predict token at offset k+1.
    All K losses are summed (equal weight) alongside the main next-token loss.
    This gives the model 2–4× more gradient signal per forward pass.

    During inference, the K heads act as a draft model for speculative
    decoding: the main model verifies K tokens in one forward pass, accepting
    any suffix where the draft matches. This yields 2–4× throughput with
    zero quality loss.

    Architecture per head
    ─────────────────────
    RMSNorm → single TransformerBlock (or lightweight FFN) → RMSNorm → shared LM-head projection

    Weight sharing: the output projection reuses the main model's lm_head
    weight (set by ModernLLM after construction).

    "Better & Faster Large Language Models via Multi-token Prediction"
    """

    def __init__(self, config: ModelConfig, num_heads: int = 3) -> None:
        super().__init__()
        self.num_heads     = num_heads
        self.use_full_block = getattr(config, "mtp_use_full_block", False)

        if self.use_full_block:
            # One small transformer block + norm pair per prediction depth
            self.blocks = nn.ModuleList([
                nn.ModuleDict({
                    "norm":     RMSNorm(config.emb_dim),
                    "block":    TransformerBlock(config),
                    "out_norm": RMSNorm(config.emb_dim),
                })
                for _ in range(num_heads)
            ])
        else:
            # Lightweight MTP heads (FFN only) as per best practices
            self.ffn_layers = nn.ModuleList([
                nn.ModuleDict({
                    "norm":     RMSNorm(config.emb_dim),
                    "ffn":      SwiGLU(config.emb_dim, config.emb_dim * 2),
                    "out_norm": RMSNorm(config.emb_dim),
                })
                for _ in range(num_heads)
            ])

        # Shared projection — weight-tied to lm_head by ModernLLM
        self.proj = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        target_embeddings: torch.Tensor,  # [b, seq, emb_dim] — embeddings of tokens[:,1:]
        freqs_cis: torch.Tensor,
        logit_scale: torch.Tensor,
        mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        logits_list    = []
        h_prev         = hidden
        total_aux_loss = torch.tensor(0.0, device=hidden.device, dtype=hidden.dtype)
        seq_len        = hidden.size(1)

        for k in range(self.num_heads):
            # Shift target embeddings: head k predicts token t+k+1.
            # target_embeddings contains [Emb(t+1), Emb(t+2), ...].
            # At iteration k, we need offset k.
           if k >= target_embeddings.size(1):
               logits_list.append(
                   torch.empty(hidden.size(0), 0, self.proj.out_features,
                               device=hidden.device, dtype=hidden.dtype)
               )
               continue
    
           target         = target_embeddings[:, k : k + seq_len]
           current_h_prev = h_prev[:, :target.size(1)]

           if current_h_prev.size(1) == 0:
               logits_list.append(
                   torch.empty(hidden.size(0), 0, self.proj.out_features,
                        device=hidden.device, dtype=hidden.dtype)
               )
               continue

           h_head = current_h_prev + target

           if self.use_full_block:
               head   = self.blocks[k]
               h_head = head["norm"](h_head)
               h_head, aux = head["block"](
                   h_head,
                   freqs_cis[k + 1 : k + 1 + h_head.size(1)],
                   attention_mask=(
                       attention_mask[:, :h_head.size(1)]
                       if attention_mask is not None else None
                   ),
                   mask=(_slice_attention_mask(mask, h_head.size(1))
                         if mask is not None else None),
               )
               h_head = head["out_norm"](h_head)
           else:
               head   = self.ffn_layers[k]
               h_head = head["norm"](h_head)
               h_head = head["ffn"](h_head)
               h_head = head["out_norm"](h_head)
               aux    = torch.tensor(0.0, device=hidden.device, dtype=hidden.dtype)

           logits = self.proj(h_head) * logit_scale
           logits_list.append(logits)

           h_prev         = h_head
           total_aux_loss = total_aux_loss + aux

        return logits_list, total_aux_loss


# ══════════════════════════════════════════════════════════════════════════════
# Full Model
# ══════════════════════════════════════════════════════════════════════════════

class ModernLLM(nn.Module):
    """
    Complete decoder-only language model.

    Features
    --------
    * Weight tying  :  output projection shares weights with token embedding
                       (halves the parameter count for the embedding table)
    * RoPE buffer   :  precomputed for 2× context_length to allow longer
                       generation than training context without re-computation
    * Grad checkpoint: optional activation recomputation to save VRAM
    * KV cache      :  passed through each layer during inference

    Parameters
    ----------
    config : ModelConfig
    gradient_checkpointing : bool
        If True, recompute activations during backward instead of storing them.
        Reduces VRAM by ~30-40 % at the cost of ~30 % extra compute.
    """

    def __init__(
        self,
        config: ModelConfig,
        gradient_checkpointing: bool = False,
        mtp_heads: int | None = None,  # If None, use config.mtp_heads
        moe_aux_weight: float = 0.01,  # Load-balancing loss coefficient
    ) -> None:
        super().__init__()
        self.config              = config
        self.gradient_checkpointing = gradient_checkpointing
        self.moe_aux_weight      = moe_aux_weight

        # Use mtp_heads from config if not explicitly provided
        mtp_heads_resolved: int = (mtp_heads if mtp_heads is not None
                                   else getattr(config, "mtp_heads", 0))

        self.tok_emb  = nn.Embedding(config.vocab_size, config.emb_dim)
        self.emb_norm = RMSNorm(config.emb_dim)

        freqs_cis = precompute_freqs_cis(
            config.head_dim,
            config.context_length * 2,
            theta=getattr(config, "rope_theta", 10_000.0),
            rope_scaling_type=getattr(config, "rope_scaling_type", "linear"),
            rope_scaling_factor=getattr(config, "rope_scaling_factor", 1.0),
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        self.layers  = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm    = RMSNorm(config.emb_dim)
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

        # Use fixed logit scale if provided in config, otherwise use learnable
        # (with warning).
        logit_scale_val = getattr(config, "logit_scale", None)
        if logit_scale_val is not None:
            self.register_buffer(
                "logit_scale",
                torch.tensor(logit_scale_val, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.logit_scale = nn.Parameter(torch.ones(1) * (config.emb_dim ** -0.5))
            print("Warning: Using learnable logit_scale. Consider fixing it for stability.")

        # Multi-Token Prediction heads (optional)
        self.mtp = MultiTokenPredictionHead(config, mtp_heads_resolved) if mtp_heads_resolved > 0 else None

        # Init BEFORE weight tying (last-writer-wins on shared tensor)
        self.apply(self._init_weights)

        # Weight tying: tok_emb ↔ lm_head ↔ mtp.proj (all share one matrix)
        self.lm_head.weight = self.tok_emb.weight
        if self.mtp is not None:
            self.mtp.proj.weight = self.tok_emb.weight

    # ── Weight Initialisation ─────────────────────────────────────────────────

    def _init_weights(self, module: nn.Module) -> None:
        """
        Standard normal initialisation.
        Residual scaling is handled explicitly in TransformerBlock.forward
        via DeepNet alpha; do NOT additionally scale projection weights here
        (that would double-suppress the residual stream).
        """
        std = 0.02
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        tokens: torch.Tensor,              # [batch, seq_len]  int64
        kv_cache: KVCache | None = None,
        mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Returns a dict with keys:
          'logits'     : [b, seq, vocab]  — main next-token logits (always)
          'mtp_logits' : list of K [b, seq, vocab] — MTP head logits (training only)
          'aux_loss'   : scalar — MoE load-balancing loss, pre-weighted by
                         moe_aux_weight. Add directly to CE loss:
                         total_loss = ce_loss + out['aux_loss']

        During inference (kv_cache is not None), mtp_logits is empty and
        aux_loss is zero — no overhead.
        """
        b, seq_len = tokens.shape
        if tokens.dtype != torch.long:
            raise TypeError(f"tokens must be torch.long, got {tokens.dtype}")
        if seq_len == 0:
            raise ValueError("tokens must contain at least one token")
        if tokens.min().item() < 0 or tokens.max().item() >= self.config.vocab_size:
            raise ValueError("input token ids are out of range for the configured vocabulary")

        # Force KV cache to None during training to prevent in-place corruption
        # during gradient-checkpointing re-computation passes.
        if self.training:
            kv_cache = None

        if attention_mask is not None:
            if attention_mask.dim() != 2:
                raise ValueError(
                    f"attention_mask must have shape [batch, seq], got {tuple(attention_mask.shape)}"
                )
            if attention_mask.shape != tokens.shape:
                raise ValueError(
                    f"attention_mask shape {tuple(attention_mask.shape)} does not match "
                    f"tokens shape {tuple(tokens.shape)}"
                )

        h         = self.emb_norm(self.tok_emb(tokens))
        cache_len = kv_cache.seq_len if kv_cache is not None else 0
        if cache_len + seq_len > self.freqs_cis.size(0):
            raise ValueError(
                f"sequence length overflow: cache_len={cache_len}, seq_len={seq_len}, "
                f"max_supported={self.freqs_cis.size(0)}"
            )
        freqs_cis = self.freqs_cis[cache_len : cache_len + seq_len].to(h.device)

        total_aux_loss = torch.tensor(0.0, device=h.device, dtype=torch.float32)

        # Checkpoint wrapper that preserves keyword arguments
        def create_custom_forward(module):
            def custom_forward(*args):
                return module(*args)
            return custom_forward

        for i, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                h, aux = grad_checkpoint.checkpoint(
                    create_custom_forward(layer),
                    h, freqs_cis, i, kv_cache, mask, attention_mask,
                    use_reentrant=False,
                )
            else:
                h, aux = layer(
                    h,
                    freqs_cis,
                    layer_idx=i,
                    kv_cache=kv_cache,
                    mask=mask,
                    attention_mask=attention_mask,
                )

            total_aux_loss = total_aux_loss + aux.float()   # keep in float32

        normed = self.norm(h)

        # 5: Both branches of the original isinstance(self.logit_scale,
        # nn.Parameter) check produced identical code.  Collapsed to a single
        # unconditional assignment regardless of whether logit_scale is a
        # Parameter or a registered buffer.
        current_logit_scale = self.logit_scale.to(normed.dtype)
        logits = self.lm_head(normed) * current_logit_scale

        # ── Multi-Token Prediction (training only) ─────────────────────────
        mtp_logits = []
        if self.mtp is not None and self.training:
            target_embs = self.tok_emb(tokens[:, 1:])
            # Pad enough for all K prediction heads
            pad_size = self.mtp.num_heads
            pad = torch.zeros(
                b, pad_size, target_embs.size(-1),
                device=h.device, dtype=h.dtype,
            )
            target_embs = torch.cat([target_embs, pad], dim=1)
            mtp_logits, mtp_aux = self.mtp(
                h,
                target_embs,
                freqs_cis,
                current_logit_scale,
                mask=mask,
                attention_mask=attention_mask,
            )
            total_aux_loss = total_aux_loss + mtp_aux.float()

        return {
            "logits":     logits,
            "mtp_logits": mtp_logits,
            "aux_loss":   total_aux_loss * self.moe_aux_weight,
        }

    # ── Convenience ───────────────────────────────────────────────────────────

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Return total (or trainable-only) parameter count.

        Weight-tied parameters (tok_emb / lm_head / mtp.proj) are counted
        once; PyTorch's parameter iterator deduplicates shared tensors by id.
        """
        params = (p for p in self.parameters() if p.requires_grad or not trainable_only)
        return sum(p.numel() for p in params)
