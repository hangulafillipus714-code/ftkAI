from __future__ import annotations

import argparse
from pathlib import Path

import torch

from checkpoint.checkpoint import latest_checkpoint, load_checkpoint, load_checkpoint_metadata
from config.model_config import ModelConfig
from data_source.tokenizer import Tokenizer
from generation.generate import generate_text
from kv_cache.cache import KVCache
from model.model import ModernLLM


def build_model(checkpoint_path: str | None, tokenizer: Tokenizer, device: torch.device) -> ModernLLM:
    model_config = ModelConfig(vocab_size=tokenizer.vocab_size)
    if checkpoint_path:
        metadata = load_checkpoint_metadata(checkpoint_path, device=device)
        if metadata["model_config"]:
            model_config = ModelConfig.from_dict(metadata["model_config"])
            model_config.vocab_size = tokenizer.vocab_size

    model = ModernLLM(config=model_config).to(device)
    if checkpoint_path:
        load_checkpoint(checkpoint_path, model, device=device, strict=False)
    model.eval()
    return model


def build_chat_prompt(tokenizer: Tokenizer, prompt: str) -> str:
    normalized = prompt.strip()
    if not normalized.startswith(" "):
        normalized = f" {normalized}"
    return f"{tokenizer.user_token}\n{normalized}\n{tokenizer.assistant_token}\n "


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive prompt loop for the local model.")
    parser.add_argument("--checkpoint", default=None, help="Path to a checkpoint file.")
    parser.add_argument("--tokenizer", default="tokenizer.json", help="Path to tokenizer.json.")
    parser.add_argument("--device", default="cpu", help="Device to use, e.g. cpu or cuda.")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--mode", choices=["chat", "completion"], default="completion", 
                        help="Interaction mode: 'chat' for conversation formatting, 'completion' for raw text.")
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = Tokenizer(args.tokenizer)
    checkpoint_path = args.checkpoint or latest_checkpoint("checkpoints")
    if checkpoint_path and not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_model(checkpoint_path, tokenizer, device)
    kv_cache = KVCache(
        n_layers=model.config.n_layers,
        n_kv_heads=model.config.n_kv_heads,
        head_dim=model.config.head_dim,
        device=device,
    )

    print(f"Interactive mode ({args.mode}). Press Ctrl-D or Ctrl-C to exit.")
    while True:
        try:
            prompt = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not prompt:
            continue

        full_prompt = prompt
        if args.mode == "chat":
            full_prompt = build_chat_prompt(tokenizer, prompt)

        reply = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=full_prompt,
            max_new_tokens=args.max_new_tokens,
            context_size=model.config.context_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=device,
            kv_cache=kv_cache,
            echo=False,
        )

        if tokenizer.turn_end_token in reply:
            reply = reply.split(tokenizer.turn_end_token, 1)[0].strip()
        
        print(f"Model> {reply}")


if __name__ == "__main__":
    main()
