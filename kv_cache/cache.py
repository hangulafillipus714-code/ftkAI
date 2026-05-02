import torch
from typing import List, Dict

class BlockAllocator:
    """
    O(1) Freelist Stack allocator for KV Cache blocks.
    A true vLLM-style block manager that tracks free and allocated pages efficiently.
    """
    def __init__(self, num_blocks: int, device: torch.device):
        self.num_blocks = num_blocks
        # Freelist stack
        self.free_blocks = torch.arange(num_blocks - 1, -1, -1, device=device, dtype=torch.long)
        self.num_free = num_blocks
        
    def allocate(self, num: int) -> torch.Tensor:
        if num == 0:
            return torch.empty(0, device=self.free_blocks.device, dtype=torch.long)
        if self.num_free < num:
            raise RuntimeError(f"Out of KV cache blocks. Requested {num}, but only {self.num_free} available.")
        
        self.num_free -= num
        allocated = self.free_blocks[self.num_free : self.num_free + num]
        return allocated.flip(0)
        
    def free(self, block_ids: torch.Tensor) -> None:
        num = block_ids.size(0)
        if num == 0:
            return
        self.free_blocks[self.num_free : self.num_free + num] = block_ids
        self.num_free += num


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
    max_batch_size : Maximum number of simultaneous sequences.
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
        
        self.max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
        total_blocks = max_batch_size * self.max_blocks_per_seq
        self.allocator = BlockAllocator(total_blocks, device)
        
        # Physical Cache [num_blocks, n_layers, 2 (K/V), n_kv_heads, block_size, head_dim]
        self.physical_cache = torch.zeros(
            total_blocks, n_layers, 2, n_kv_heads, block_size, head_dim,
            device=device, dtype=dtype
        )
        
        # Block tables: [max_batch_size, max_blocks_per_seq] -> physical block ID
        self.block_tables = torch.full((max_batch_size, self.max_blocks_per_seq), -1, device=device, dtype=torch.long)
        
        # Seq lengths: [max_batch_size]
        self.seq_lengths = torch.zeros(max_batch_size, device=device, dtype=torch.long)
        
        self.max_batch_size = max_batch_size

    def update(
        self,
        layer_idx: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> None:
        """
        Vectorized update of the physical cache without python loops.
        k_new: [batch_size, n_kv_heads, new_seq, head_dim]
        """
        batch_size = k_new.shape[0]
        new_seq = k_new.shape[2]
        
        if batch_size > self.max_batch_size:
            raise ValueError(f"Batch size {batch_size} exceeds max_batch_size {self.max_batch_size}")
            
        # 1. Allocate blocks if needed (only on layer 0 to keep synchronization)
        if layer_idx == 0:
            current_lens = self.seq_lengths[:batch_size]
            current_blocks = (current_lens + self.block_size - 1) // self.block_size
            new_lens = current_lens + new_seq
            new_blocks = (new_lens + self.block_size - 1) // self.block_size
            
            blocks_to_allocate = new_blocks - current_blocks
            total_to_allocate = blocks_to_allocate.sum().item()
            
            if total_to_allocate > 0:
                allocated = self.allocator.allocate(total_to_allocate)
                # Scatter allocated blocks into block_tables vectorially
                b_indices = torch.arange(batch_size, device=self.device).repeat_interleave(blocks_to_allocate)
                
                max_alloc = blocks_to_allocate.max().item()
                if max_alloc > 0:
                    mask = torch.arange(max_alloc, device=self.device).unsqueeze(0) < blocks_to_allocate.unsqueeze(1)
                    block_offsets = current_blocks.unsqueeze(1) + torch.arange(max_alloc, device=self.device).unsqueeze(0)
                    valid_offsets = block_offsets[mask]
                    self.block_tables[b_indices, valid_offsets] = allocated

        # 2. Write new tokens into physical paged blocks using vectorization
        current_lens = self.seq_lengths[:batch_size]
        
        # Create token indices: [batch_size, new_seq]
        seq_indices = current_lens.unsqueeze(1) + torch.arange(new_seq, device=self.device).unsqueeze(0)
        
        logical_block_idx = seq_indices // self.block_size
        block_offset = seq_indices % self.block_size
        
        # Get physical block IDs: [batch_size, new_seq]
        physical_block_ids = self.block_tables[:batch_size].gather(1, logical_block_idx)
        
        # Transpose to [batch_size, new_seq, n_kv_heads, head_dim]
        k_write = k_new.transpose(1, 2)
        v_write = v_new.transpose(1, 2)
        
        # Write directly via advanced indexing (zero Python loops)
        self.physical_cache[physical_block_ids, layer_idx, 0, :, block_offset, :] = k_write.to(self.dtype)
        self.physical_cache[physical_block_ids, layer_idx, 1, :, block_offset, :] = v_write.to(self.dtype)
        
        # 3. Update seq_lengths after the last layer
        if layer_idx == self.n_layers - 1:
            self.seq_lengths[:batch_size] += new_seq

    def reset(self) -> None:
        """
        Frees all allocated physical blocks back to the block manager (O(1)).
        """
        allocated_mask = self.block_tables != -1
        allocated_blocks = self.block_tables[allocated_mask]
        if allocated_blocks.numel() > 0:
            self.allocator.free(allocated_blocks)
        self.block_tables.fill_(-1)
        self.seq_lengths.fill_(0)

    @property
    def seq_len(self) -> int:
        return self.seq_lengths[0].item() if self.seq_lengths.numel() > 0 else 0

    def __repr__(self) -> str:
        used_blocks = self.allocator.num_blocks - self.allocator.num_free
        return (
            f"PagedKVCache(n_layers={self.n_layers}, "
            f"blocks_used={used_blocks}/{self.allocator.num_blocks}, "
            f"cached_len={self.seq_len})"
        )