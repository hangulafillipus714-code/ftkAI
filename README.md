# ftkAI

`ftkAI` is a local decoder-only LLM project built around a LLaMA-style transformer, a custom tokenizer/data pipeline, checkpointed training, and optional PRM-backed reasoning memory.

This repository now has two documentation layers:

- `README.md` at the repo root explains the base model, training, data pipeline, and inference flow.
- `PRM/readme.md` explains the Persistent Reasoning Memory subsystem and how it is used with the base model.

## What Is In This Repo

Core training and inference:

- `model/` – the `ModernLLM` architecture
- `train/` – training entrypoint
- `generation/` – token generation helpers
- `data_source/` – tokenizer wrapper, dataset building, streaming loaders, quality filtering
- `checkpoint/` – checkpoint save/load utilities
- `optimizer/`, `scheduler/`, `distributed/`, `kv_cache/`, `utils/` – support modules

Optional reasoning memory:

- `PRM/` – persistent reasoning memory engine
- `prm_generate.py` – example generation path using PRM

User-facing scripts:

- `train/train.py` – train the model
- `generate.py` – one-shot generation
- `ask_model.py` – interactive local prompting
- `prm_generate.py` – PRM-backed prompting

## Base Model

The model in `model/model.py` is a decoder-only transformer with:

- RMSNorm
- RoPE
- Grouped-query attention
- SwiGLU feed-forward blocks
- optional sparse MoE
- optional multi-token prediction heads
- KV-cache support for inference

The project is set up for checkpoint-based training and local generation, not as a packaged inference service.

## Data Pipeline

The repository supports two data-loading modes.

Standard map-style loading:

- good for smaller and medium local datasets
- supports curriculum sampling
- loads source files into memory during dataset construction

Streaming loading:

- intended for Kaggle-scale or large uploaded corpora
- uses `IterableDataset`
- supports raw text and `.jsonl`
- does not support top-level `.json` `entries` datasets
- does not support curriculum sampling
- requires `max_steps` to be set explicitly in training

The training config controls this through `TrainConfig.use_streaming_dataset`.

## Quality Filtering

The data pipeline includes a conservative quality filter in `data_source/dataset.py`.

It is designed to remove obvious garbage without aggressively deleting useful examples. It rejects data such as:

- symbol soup
- repeated spam lines
- extremely low-information text
- malformed or empty prompt/completion rows

It is intentionally not a semantic quality ranker. It will not detect every weak example, but it should avoid overfiltering normal text.

Filter behavior is controlled from `TrainConfig`:

- `use_quality_filter`
- `quality_min_chars`
- `quality_min_alpha_ratio`
- `quality_max_symbol_ratio`
- `quality_max_repeated_line_fraction`
- `quality_min_unique_token_ratio`

## Training

Main entrypoint:

```bash
python train/train.py
```

Multi-GPU:

```bash
torchrun --nproc_per_node=8 train/train.py
```

Training behavior is defined by `config/train_config.py` and `config/model_config.py`.

Current training stack includes:

- AMP support
- DDP support
- gradient checkpointing
- scheduler selection
- optimizer selection
- checkpoint resume
- validation/eval loop
- atomic checkpoint writes
- best-checkpoint saving
- streaming-safe data loading

Important operational notes:

- if `use_streaming_dataset=True`, set `max_steps`
- if `use_streaming_dataset=True`, do not use curriculum batching
- if you use `.jsonl`, rows should contain `prompt` and `completion`
- if you use `.json`, the expected shape is a top-level `entries` list with `question` and `final_answer`

## Inference

One-shot generation:

```bash
python generate.py --prompt "Wa lele po?" --device cpu
```

Interactive prompt loop:

```bash
python ask_model.py --device cpu --mode chat
```

These scripts:

- load the latest checkpoint by default from `checkpoints/`
- rebuild the model config from checkpoint metadata when available
- use `tokenizer.json`
- run local generation through `generation/generate.py`

## How PRM Fits In

PRM is not the base training loop. It is an optional reasoning-memory layer that can wrap model usage for long-running task work.

Use the base project when you want:

- normal model training
- checkpointed local inference
- tokenizer and data pipeline work

Use PRM when you want:

- persistent project/task memory
- state tracking across steps
- memory-aware prompting
- optional refinement/correction loops

See [PRM/readme.md](PRM/readme.md) for PRM-specific usage.

## Typical Workflows

Train the base model:

```bash
python train/train.py
```

Generate from the latest checkpoint:

```bash
python generate.py --prompt "Explain gravity."
```

Open an interactive local prompt session:

```bash
python ask_model.py --mode chat
```

Run PRM-backed generation:

```bash
python prm_generate.py --project "My Project" --prompt "Implement GQA"
```

## Status

The base model/training code and the PRM code live in the same repository, but they are not the same subsystem.

- `ftkAI` is the base LLM project
- `PRM` is an optional memory/control layer on top of it

The base scripts `train/train.py`, `generate.py`, and `ask_model.py` do not automatically use PRM. PRM is used explicitly through `prm_generate.py` or through direct integration from `PRM/integration.py`.
