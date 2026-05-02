"""
generation/generate.py
----------------------
Autoregressive token generation with KV-cache, temperature, top-k, and top-p.

Generation algorithm
────────────────────
1. Encode the prompt into token IDs.
2. On the first forward pass, feed the entire prompt so the KV cache is
   populated for all prompt tokens.
3. On each subsequent step, feed only the single newly-generated token;
   the cache provides all previous context.
4. Apply temperature scaling, then top-k / top-p (nucleus) filtering.
5. Sample the next token and append to the sequence.
6. Stop at max_new_tokens or when the EOS token is generated.
"""

import torch
import torch.nn.functional as F

from kv_cache.cache import KVCache


def _extract_logits(model_output: torch.Tensor | dict) -> torch.Tensor:
    if isinstance(model_output, dict):
        return model_output["logits"]
    return model_output


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    idx: torch.Tensor,              # [batch, prompt_len]  int64
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,     # nucleus sampling (0.9 is a common value)
    repetition_penalty: float = 1.0, # penalise re-generating recent tokens
    eos_token_id: int | None = None,
    kv_cache: KVCache | None = None,
) -> torch.Tensor:
    """
    Autoregressively generate `max_new_tokens` tokens from a prompt.

    Parameters
    ----------
    model          : The language model (eval mode; no grad).
    idx            : Prompt token IDs  [batch, prompt_len].
    max_new_tokens : Number of tokens to generate.
    context_size   : Maximum sequence length the model was trained on.
                     Used to truncate the prompt on the first pass if needed.
    temperature    : Softmax temperature.
                     1.0 = unchanged, <1.0 = sharper, >1.0 = flatter.
                     0.0 = greedy (argmax), no sampling.
    top_k          : If set, keep only the top-k logits before sampling.
    top_p          : If set, keep the smallest set of tokens whose cumulative
                     probability exceeds top_p (nucleus sampling).
    repetition_penalty : Penalise re-generating tokens that have recently
                         appeared in the sequence. 1.0 = no penalty.
                         >1.0 discourages repetition, <1.0 encourages it.
    eos_token_id   : If set, stop generation when this token is produced.
    kv_cache       : If provided, use incremental decoding (much faster).
                     The cache is reset before generation starts so it is
                     safe to reuse the same KVCache object across calls.

    Returns
    -------
    torch.Tensor  :  [batch, prompt_len + max_new_tokens]  int64
    """
    model.eval()

    # Reset cache so prior state doesn't bleed into this generation call
    if kv_cache is not None:
        kv_cache.reset()

    prompt_len = idx.shape[1]

    for token_idx in range(max_new_tokens):

        # ── Select input for this step ─────────────────────────────────────────
        if kv_cache is not None and token_idx > 0:
            # Cache is warm: only feed the single newest token
            idx_cond = idx[:, -1:]
        else:
            # First step: feed the whole prompt (truncated to context_size)
            idx_cond = idx[:, -context_size:]

        # ── Forward pass ───────────────────────────────────────────────────────
        logits = _extract_logits(model(idx_cond, kv_cache=kv_cache))[:, -1, :]  # [batch, vocab]

        # ── Repetition penalty ─────────────────────────────────────────────────
        if repetition_penalty != 1.0:
            # Create a penalty mask for tokens already in the sequence
            # We iterate through the batch to apply penalty to each sequence individually
            for i in range(idx.size(0)): # Iterate over batch dimension
                # Get unique tokens in the current sequence (excluding the current token if it's part of the prompt)
                past_tokens = torch.unique(idx[i])
                # Apply penalty to logits
                # If logit > 0, penalise by division. If logit < 0, penalise by multiplication.
                # This prevents very low negative logits from becoming less negative (more likely).
                for t in past_tokens:
                    if logits[i, t] > 0:
                        logits[i, t] /= repetition_penalty
                    else:
                        logits[i, t] *= repetition_penalty

        # ── Greedy decode ──────────────────────────────────────────────────────
        if temperature == 0.0:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        else:
            # ── Temperature scaling ────────────────────────────────────────────
            logits = logits / temperature

            # ── Top-k filtering ────────────────────────────────────────────────
            if top_k is not None and top_k > 0:
                k = min(top_k, logits.size(-1))
                top_values, _ = torch.topk(logits, k, dim=-1)
                # Mask everything below the k-th value
                logits = logits.masked_fill(
                    logits < top_values[:, [-1]], float("-inf")
                )

            # ── Top-p (nucleus) filtering ──────────────────────────────────────
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(probs, dim=-1)
                
                # Remove tokens whose cumulative probability exceeds top_p
                # We keep tokens where cumulative_probs - probs <= top_p
                remove_mask = (cumulative_probs - probs) > top_p
                sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))
                
                # Scatter back to original order, using -inf as default
                logits = torch.full_like(logits, float("-inf")).scatter_(
                    dim=-1, index=sorted_indices, src=sorted_logits
                )

            # ── Sample ────────────────────────────────────────────────────────
            probs    = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # [batch, 1]

        # ── Append to sequence ─────────────────────────────────────────────────
        idx = torch.cat([idx, idx_next], dim=1)

        # ── Early stop on EOS ──────────────────────────────────────────────────
        if eos_token_id is not None and (idx_next == eos_token_id).all():
            break

    return idx


def generate_text(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    context_size: int = 1024,
    temperature: float = 1.0,
    top_k: int | None = 50,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    device: torch.device = torch.device("cpu"),
    kv_cache: KVCache | None = None,
    echo: bool = False,
) -> str:
    """
    Convenience wrapper: encode a text prompt, generate, decode back to text.

    Parameters
    ----------
    (see `generate` for shared parameters)
    tokenizer : Tokenizer instance with .encode() and .decode() methods.
    prompt    : Raw text prompt string.
    device    : Device to put the prompt tensor on.
    echo      : If True, returned string includes the prompt.

    Returns
    -------
    str  :  Decoded text (generated continuation, optionally with prompt).
    """
    if not prompt:
        prompt = tokenizer.bos_token

    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        token_ids = [tokenizer.bos_token_id]

    if hasattr(model, "config"):
        context_size = min(context_size, model.config.context_length)

    encoded = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)

    output = generate(
        model=model,
        idx=encoded,
        max_new_tokens=max_new_tokens,
        context_size=context_size,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eot_token_id,
        kv_cache=kv_cache,
    )

    if echo:
        return tokenizer.decode(output.squeeze(0).tolist(), skip_special_tokens=True)
    else:
        # Return only the new tokens
        return tokenizer.decode(output.squeeze(0)[len(token_ids):].tolist(), skip_special_tokens=True)
