"""
data_source/build_tokenizer.py
------------------------------
Industrial LLM Tokenizer Training script.

Builds a robust Byte-Pair Encoding (BPE) model dynamically across massive 
datasets avoiding Out Of Memory (OOM) states with native iterators.

Configured strictly to cap vocab dimensions at 64,000 for symmetric 
tensor calculations representing the 1.5 Billion architecture.
"""

import os
import json
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer, decoders, pre_tokenizers, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

# ==============================================================================
# Token Standards
# ==============================================================================

# Map strictly to the identical special tokens expected by `tokenizer.py` wrappers
SPECIAL_TOKENS = [
    "[PAD]",
    "[UNK]",
    "<|bos|>",
    "<|eot|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
]

VOCAB_SIZE_TARGET = 64000
MIN_FREQUENCY = 2  # Strip isolated 1-time typing errors/noise


def build_streaming_iterator(file_paths: list[str]) -> Iterator[str]:
    """
    Safely yields continuous text blocks from mixed format sources 
    without loading entire multi-gigabyte structures into active RAM!
    """
    for file_path in file_paths:
        p = Path(file_path)
        if not p.exists():
            print(f"[Warning] Skipping missing source data: {p.name}")
            continue

        print(f"[*] Streaming text arrays from: {p.name}")
        
        if p.suffix == ".json":
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "entries" in data:
                    for entry in data["entries"]:
                        q = str(entry.get("question", "")).strip()
                        a = str(entry.get("final_answer", "")).strip()
                        if q or a:
                            yield q + "\n" + a
                            
        elif p.suffix == ".jsonl":
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        yield str(obj.get("prompt", "")) + "\n" + str(obj.get("completion", ""))
                    except:
                        pass
        else:
            # Assumed raw text corpus, stream line safely chunked 
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    cleaned_line = line.strip()
                    if cleaned_line:
                        yield cleaned_line


def train_production_tokenizer(input_paths: list[str], output_json_name: str = "tokenizer.json"):
    """
    Bootstraps the tokenizer configurations and explicitly drives BPE trainers globally over the streaming text lines!
    """
    # 1. Initialize empty BPE configuration ensuring unknown properties map reliably
    bpe_model = BPE(unk_token="[UNK]")
    tokenizer = Tokenizer(bpe_model)
    
    # 2. Add structural formatting (Crucial for python indentations/tabs logic)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    # 3. Formulate the explicit constrained Trainer protocol
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE_TARGET,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    print("-" * 50)
    print(f"Bootstrapping BPE Tokenizer -> Targeted Nodes: {VOCAB_SIZE_TARGET:,}")
    print("-" * 50)
    
    # 4. Trigger the intense multi-threaded iteration scan!
    streamer = build_streaming_iterator(input_paths)
    tokenizer.train_from_iterator(streamer, trainer=trainer)

    # 5. Flush state aggressively onto disk ready for runtime processing
    output_path = Path(__file__).parent.parent / output_json_name
    tokenizer.save(str(output_path))
    
    print("-" * 50)
    print(f"[SUCCESS] Tokenizer completely built!")
    print(f"[SUCCESS] Total Vocab Indexed: {tokenizer.get_vocab_size():,}")
    print(f"[SUCCESS] Saved directly to {output_path.absolute()}")
    print("-" * 50)


if __name__ == "__main__":
    # Standard location defaults, adjustable per cloud environment paths
    # Will pull identically from standard dataset structures
    test_paths = [
        os.path.join(os.path.dirname(__file__), "..", "DATA SPOT", "training_data.txt")
    ]
    
    # Check if we should execute 
    train_production_tokenizer(input_paths=test_paths)
