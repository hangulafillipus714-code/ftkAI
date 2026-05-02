
import os
import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.model_config import ModelConfig
from config.train_config import TrainConfig
from data_source.tokenizer import Tokenizer
from data_source.dataset import create_dataloader_from_paths
from checkpoint.checkpoint import latest_checkpoint, load_checkpoint_metadata

print("Step 1: Init Config")
t_cfg = TrainConfig()
m_cfg = ModelConfig()

print("Step 2: Init Tokenizer")
tokenizer = Tokenizer(t_cfg.tokenizer_path)
print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

print("Step 3: Check Checkpoints")
ckpt_path = latest_checkpoint(t_cfg.checkpoint_dir)
print(f"Latest checkpoint: {ckpt_path}")

if ckpt_path:
    print("Step 4: Load Checkpoint Metadata")
    ckpt_meta = load_checkpoint_metadata(ckpt_path, device=torch.device("cpu"))
    print(f"Metadata loaded: {ckpt_meta.keys()}")

print("Step 5: Init Model")
model = ModernLLM(
    config=m_cfg,
    gradient_checkpointing=t_cfg.gradient_checkpointing,
    mtp_heads=m_cfg.mtp_heads,
).to(torch.device("cpu"))
print(f"Model initialized with {model.num_parameters():,} parameters")

print("Step 6: Create Dataloader")
dataloader = create_dataloader_from_paths(
    data_paths=(t_cfg.data_path,),
    tokenizer=tokenizer,
    batch_size=t_cfg.batch_size,
    max_length=t_cfg.max_length,
    stride=t_cfg.stride,
    shuffle=t_cfg.shuffle,
    num_workers=t_cfg.num_workers,
    use_curriculum=t_cfg.use_curriculum,
    use_quality_filter=False, # Disable to avoid the bug I found
)
print("Dataloader created")

print("Step 6: Iter Dataloader")
for i, batch in enumerate(dataloader):
    print(f"Batch {i} keys: {batch.keys()}")
    if i >= 2:
        break
print("Finished test")
