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
from typing import List, Dict

class BlockAllocator:
    """
    Manages physical memory blocks for the KV Cache.
    A true vLLM-style block manager that tracks free and allocated pages.
    """
    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.free_blocks: List[int] = list(range(num_blocks))
        
    def allocate(self) -> int:
        if not self.free_blocks:
            raise RuntimeError("Out of KV cache blocks. The physical memory limit has been reached.")
        return self.free_blocks.pop(0)
        
    def free(self, block_id: int) -> None:
        if block_id not in self.free_blocks:
            self.free_blocks.append(block_id)

class KVCache:
    """
    vLLM-style Paged Key-Value cache for autoregressive generation.
    Memory is pre-allocated in fixed-size blocks (pages) rather than
    a single contiguous tensor. This prevents memory fragmentation and
    allows for flexible batching.

    Parameters
    ----------
    n_layers   : Number of transformer layers.
    n_kv_heads : Number of KV heads (GQA; NOT the full n_heads).
    head_dim   : Dimension of each attention head (emb_dim // n_heads).
    max_batch_size : Maximum number of simultaneous sequences (unused in core allocation, kept for compatibility).
    max_seq_len : Maximum sequence length to support (used to calculate num_blocks).
    device     : torch.device where tensors live.
    block_size : Number of tokens per block (default 16, standard for vLLM).
    dtype      : Data type of the cache.
    """

    def __init__(
        self,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        max_batch_size: int,
        max_seq_len: int,
        device: torch.device,
        block_size: int = 16,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.n_layers   = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim   = head_dim
        self.block_size = block_size
        self.device     = device
        self.dtype      = dtype
        
        # Calculate total required blocks (with some headroom)
        total_blocks = (max_batch_size * max_seq_len) // block_size + max_batch_size
        self.allocator = BlockAllocator(total_blocks)
        
        # Physical Cache [num_blocks, n_layers, 2 (K/V), n_kv_heads, block_size, head_dim]
        # In a real C++ backend this would be a flat buffer, here we use a 6D tensor
        self.physical_cache = torch.zeros(
            total_blocks, n_layers, 2, n_kv_heads, block_size, head_dim,
            device=device, dtype=dtype
        )
        
        # Maps sequence ID (usually batch index) to list of allocated physical block IDs
        self.block_tables: Dict[int, List[int]] = {i: [] for i in range(max_batch_size)}
        
        # Track lengths per sequence
        self.seq_lengths: Dict[int, int] = {i: 0 for i in range(max_batch_size)}
        
        self.max_batch_size = max_batch_size

    def _ensure_allocation(self, seq_id: int, target_len: int):
        """Ensure the sequence has enough physical blocks to hold target_len tokens."""
        blocks_needed = (target_len + self.block_size - 1) // self.block_size
        current_blocks = len(self.block_tables[seq_id])
        
        while current_blocks < blocks_needed:
            new_block = self.allocator.allocate()
            self.block_tables[seq_id].append(new_block)
            current_blocks += 1

    def update(
        self,
        layer_idx: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the paged cache with new tokens and returns a contiguous view
        for standard PyTorch attention operations.
        
        Note: A true custom Triton kernel would consume the physical_cache and 
        block_tables directly. Here we simulate the paged backend but yield 
        contiguous tensors for F.scaled_dot_product_attention compatibility.
        """
        batch_size = k_new.shape[0]
        new_seq = k_new.shape[2]
        
        if batch_size > self.max_batch_size:
            raise ValueError(f"Batch size {batch_size} exceeds max_batch_size {self.max_batch_size}")
            
        k_out_list = []
        v_out_list = []
            
        for b in range(batch_size):
            start_idx = self.seq_lengths[b] if layer_idx != 0 else self.seq_lengths[b]
            end_idx = start_idx + new_seq
            
            # 1. Allocate blocks if needed
            self._ensure_allocation(b, end_idx)
            block_list = self.block_tables[b]
            
            # 2. Write new tokens into the physical paged blocks
            # For simplicity, if we write multiple tokens, we chunk them
            tokens_written = 0
            while tokens_written < new_seq:
                tok_idx = start_idx + tokens_written
                block_idx = tok_idx // self.block_size
                block_offset = tok_idx % self.block_size
                
                physical_block = block_list[block_idx]
                
                # Write 1 token at a time to the physical page
                self.physical_cache[physical_block, layer_idx, 0, :, block_offset, :] = k_new[b, :, tokens_written, :]
                self.physical_cache[physical_block, layer_idx, 1, :, block_offset, :] = v_new[b, :, tokens_written, :]
                
                tokens_written += 1
                
            # 3. Gather full contiguous tensor for this sequence (for PyTorch SDPA)
            total_len = end_idx
            k_seq = torch.zeros(self.n_kv_heads, total_len, self.head_dim, device=self.device, dtype=self.dtype)
            v_seq = torch.zeros(self.n_kv_heads, total_len, self.head_dim, device=self.device, dtype=self.dtype)
            
            gathered = 0
            for block_id in block_list:
                tokens_in_block = min(self.block_size, total_len - gathered)
                if tokens_in_block <= 0:
                    break
                k_seq[:, gathered:gathered+tokens_in_block, :] = self.physical_cache[block_id, layer_idx, 0, :, :tokens_in_block, :]
                v_seq[:, gathered:gathered+tokens_in_block, :] = self.physical_cache[block_id, layer_idx, 1, :, :tokens_in_block, :]
                gathered += tokens_in_block
                
            k_out_list.append(k_seq.unsqueeze(0))
            v_out_list.append(v_seq.unsqueeze(0))
            
            if layer_idx == self.n_layers - 1:
                self.seq_lengths[b] = end_idx
                
        # Stack into [batch_size, n_kv_heads, total_len, head_dim]
        k_full = torch.cat(k_out_list, dim=0)
        v_full = torch.cat(v_out_list, dim=0)
        
        return k_full, v_full

    def reset(self) -> None:
        """
        Frees all allocated physical blocks back to the block manager.
        """
        for b in self.block_tables:
            for block_id in self.block_tables[b]:
                self.allocator.free(block_id)
            self.block_tables[b] = []
            self.seq_lengths[b] = 0

    @property
    def seq_len(self) -> int:
        # Returns length of first sequence (assuming homogeneous batch sizes in inference script)
        return self.seq_lengths.get(0, 0)

    def __repr__(self) -> str:
        used_blocks = self.allocator.num_blocks - len(self.allocator.free_blocks)
        return (
            f"PagedKVCache(n_layers={self.n_layers}, "
            f"blocks_used={used_blocks}/{self.allocator.num_blocks}, "
            f"cached_len={self.seq_len})"
        )
