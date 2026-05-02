"""
kv_cache/cache.py
-----------------
Key-Value cache for efficient autoregressive inference.

During generation each new token only needs to attend to all *previous*
keys and values.  Recomputing them at every step is O(n²) work.  The cache
stores them so each step is O(n) instead.

Memory layout per layer (after update):
    k / v shape:  [batch, n_kv_heads, total_cached_len, head_dim]

Important design choices
------------------------
* The cache stores tensors at n_kv_heads resolution (before GQA expansion).
  GQA expansion (repeat_interleave) happens inside the attention module AFTER
  the cache update, keeping storage proportional to n_kv_heads not n_heads.

* RoPE positions are tracked via the cached length so the model can slice
  freqs_cis correctly: freqs_cis[cache_len : cache_len + new_seq_len].

* Batch-size mismatch auto-reinitialises the slot (handles re-use across
  different generation calls without manually resetting).
"""

import torch


class KVCache:
    """
    Per-layer key-value cache for autoregressive generation.

    Parameters
    ----------
    n_layers   : Number of transformer layers.
    n_kv_heads : Number of KV heads (GQA; NOT the full n_heads).
    head_dim   : Dimension of each attention head  (emb_dim // n_heads).
    device     : torch.device where tensors live.
    """

    def __init__(
        self,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        device: torch.device,
    ) -> None:
        self.n_layers   = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim   = head_dim
        self.device     = device

        # Initialise with zero-length sequence dimension so the first update
        # simply concatenates onto an empty tensor.
        self._k: list[torch.Tensor] = self._empty_list()
        self._v: list[torch.Tensor] = self._empty_list()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _empty_list(self) -> list[torch.Tensor]:
        """Create a list of empty (batch=1) tensors, one per layer."""
        return [
            torch.zeros(1, self.n_kv_heads, 0, self.head_dim, device=self.device)
            for _ in range(self.n_layers)
        ]

    def _reinit_layer(self, layer_idx: int, batch_size: int) -> None:
        """Reset a single layer's cache for a new batch size."""
        self._k[layer_idx] = torch.zeros(
            batch_size, self.n_kv_heads, 0, self.head_dim, device=self.device
        )
        self._v[layer_idx] = torch.zeros(
            batch_size, self.n_kv_heads, 0, self.head_dim, device=self.device
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        layer_idx: int,
        k_new: torch.Tensor,   # [batch, n_kv_heads, new_seq, head_dim]
        v_new: torch.Tensor,   # [batch, n_kv_heads, new_seq, head_dim]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Append new keys/values to the cache for `layer_idx` and return the
        full accumulated key/value tensors.

        Returns
        -------
        k_full, v_full  :  [batch, n_kv_heads, total_len, head_dim]
        """
        batch_size = k_new.shape[0]

        # Auto-reinitialise if batch size changed (e.g. first call after reset)
        if self._k[layer_idx].shape[0] != batch_size:
            self._reinit_layer(layer_idx, batch_size)

        self._k[layer_idx] = torch.cat([self._k[layer_idx], k_new], dim=2)
        self._v[layer_idx] = torch.cat([self._v[layer_idx], v_new], dim=2)

        return self._k[layer_idx], self._v[layer_idx]

    def reset(self) -> None:
        """
        Clear all cached keys and values (keep batch size / device).
        Call between independent generation requests.
        """
        self._k = [t[:, :, :0, :] for t in self._k]
        self._v = [t[:, :, :0, :] for t in self._v]

    @property
    def seq_len(self) -> int:
        """Number of tokens currently stored in the cache (layer 0 as reference)."""
        return self._k[0].shape[2]

    # Expose k/v as properties so attention can access them by index
    @property
    def k(self) -> list[torch.Tensor]:
        return self._k

    @property
    def v(self) -> list[torch.Tensor]:
        return self._v

    def __repr__(self) -> str:
        return (
            f"KVCache(n_layers={self.n_layers}, "
            f"n_kv_heads={self.n_kv_heads}, "
            f"head_dim={self.head_dim}, "
            f"cached_len={self.seq_len})"
        )
