#!/bin/bash
# kaggle_bootstrap.sh — Master automation for Kaggle T4*2 Environments
# Sets up dependencies, builds the 64k vocab, and launches Distributed Training.

set -e

echo "🚀 Starting FTK-AI Kaggle Bootstrap..."

# 1. Install missing dependencies
echo "[1/3] Installing industrial-grade requirements..."
pip install tokenizers pydantic torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 -q

# 2. Build the Tokenizer (64k Vocab)
# This is required because the 1.5B model configuration needs a specific vocab embedding size.
echo "[2/3] Building 64k BPE Tokenizer from dataset..."
if [ -f "DATA SPOT/training_data.txt" ]; then
    python3 data_source/build_tokenizer.py
else
    echo "⚠️ Warning: DATA SPOT/training_data.txt not found. Using existing tokenizer.json if available."
fi

# 3. Launch Distributed Training (2 GPUs)
echo "[3/3] Launching Distributed Training (DDP) on 2x T4 GPUs..."
# We use torchrun to automatically handle rank and world size assignment
torchrun --nproc_per_node=2 train/train.py \
    --use_amp True \
    --amp_dtype float16 \
    --optimizer lion \
    --batch_size 2 \
    --grad_accum_steps 16 \
    --gradient_checkpointing True

echo "✅ Training session initiated."
