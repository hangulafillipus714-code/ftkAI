"""
prm_generate.py
---------------
Reasoning-Aware Generation Interface for ModernLLM.
Uses the Persistent Reasoning Memory (PRM) to maintain project state and 
implements automated self-correction loops.

Usage:
  python3 prm_generate.py --project "My Project" --prompt "Implement GQA"
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

# Ensure local imports work
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from PRM import PRMStore, PRMController, load_or_create
from data_source.tokenizer import Tokenizer
from kv_cache.cache import KVCache
from model.model import ModernLLM
from checkpoint.checkpoint import latest_checkpoint, load_checkpoint
from config.model_config import ModelConfig
from generation.generate import generate_text

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("PRM-Gen")

def main():
    parser = argparse.ArgumentParser(description="Cognitive Generation with PRM")
    parser.add_argument("--prompt", required=True, help="Your instruction")
    parser.add_argument("--project", default="production_project", help="PRM Project Name")
    parser.add_argument("--db", default="./prm_memory.db", help="SQL Store Path")
    parser.add_argument("--checkpoint", default=None, help="Path to weights")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--refine", action="store_true", help="Enable self-correction loop")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "auto" else args.device)
    
    # 1. Load Tokenizer & Model
    tokenizer = Tokenizer("tokenizer.json")
    ckpt = args.checkpoint or latest_checkpoint("checkpoints")
    
    # Ensure config matches current production 1.5B specs
    m_cfg = ModelConfig(vocab_size=tokenizer.vocab_size)
    model = ModernLLM(config=m_cfg).to(device)
    
    if ckpt:
        load_checkpoint(ckpt, model, device=device, strict=False)
    model.eval()

    # 2. Setup model wrapper function for PRM
    kv_cache = KVCache(
        n_layers=model.config.n_layers,
        n_kv_heads=model.config.n_kv_heads,
        head_dim=model.config.head_dim,
        max_batch_size=1,
        max_seq_len=model.config.context_length,
        device=device,
    )
    def model_fn(prompt_str: str) -> str:
        # Simple generation wrapper
        with torch.no_grad():
            output = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_str,
                max_new_tokens=256,
                temperature=0.7,
                device=device,
                kv_cache=kv_cache,
            )
        return output

    # 3. Setup PRM Controller
    store = PRMStore(args.db)
    state = load_or_create(store, args.project, goal="Industrial Task Completion")
    
    ctrl = PRMController(
        model_fn=model_fn,
        store=store,
        project_id=state.project_id
    )

    # 4. Execute with Cognitive Loop
    print(f"\n[PRM] Project: {state.project_name} | Goal: {state.goal}")
    print("-" * 50)
    
    if args.refine:
        logger.info("Executing with Self-Correction Refinement...")
        result = ctrl.step_with_refinement(args.prompt, max_refinement_steps=1)
    else:
        result = ctrl.step(args.prompt)

    print("\n=== Model Output ===")
    print(result.model_output)
    
    print("\n=== Current PRM Status ===")
    status = ctrl.status()
    print(f"Stage: {status['stage']}")
    print(f"Goal Confidence: {status.get('goal_confidence', 0.0):.2%}")
    if status.get('next_actions'):
        print(f"Next Actions: {status['next_actions']}")
    
    store.close()

if __name__ == "__main__":
    main()
