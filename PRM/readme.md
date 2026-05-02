# PRM

`PRM` stands for Persistent Reasoning Memory.

It is an optional subsystem inside `ftkAI` for maintaining long-lived project state across multiple model calls. Its job is not to replace the base model. Its job is to help the model retain task structure, working context, and next actions over longer sessions.

If you have not read the root project doc yet, start with [../README.md](../README.md).

## What PRM Is

PRM is a memory and control layer built around:

- project state storage
- compressed memory summaries
- task and stage tracking
- memory-aware prompting
- optional self-correction/refinement loops

Relevant modules:

- `store.py` – persistence layer
- `schema.py` – structured state definitions
- `controller.py` – orchestration and step execution
- `compressor.py` – memory compression logic
- `integration.py` – model integration helpers
- `curriculum.py` – curriculum utilities used separately from the main streaming training path

## What PRM Is Not

PRM is not:

- the transformer model itself
- the default training loop
- a replacement for checkpointing
- automatically active in `train/train.py`, `generate.py`, or `ask_model.py`

Those base scripts run without PRM unless you explicitly wrap them.

## How PRM Is Used

There are two practical ways to use it in this repository.

### 1. PRM-backed generation script

The simplest entrypoint already present is:

```bash
python prm_generate.py --project "My Project" --prompt "Implement GQA"
```

That script:

- loads the tokenizer and model
- loads a checkpoint if available
- creates or reuses a PRM project state
- wraps generation through `PRMController`
- optionally runs a refinement step when `--refine` is enabled

This is the easiest way to use PRM today.

### 2. Direct integration in your own code

If you already have a model function, you can wire PRM through the controller and store yourself.

Typical shape:

```python
from PRM import PRMStore, PRMController, load_or_create

store = PRMStore("./prm_memory.db")
state = load_or_create(store, "my_project", goal="Finish the task")

controller = PRMController(
    model_fn=my_generation_function,
    store=store,
    project_id=state.project_id,
)

result = controller.step("Implement the next part of the feature")
```

## Relationship To ftkAI

The split is:

- `ftkAI` provides the model, tokenizer, data pipeline, training loop, checkpoints, and generation utilities
- `PRM` provides persistent task memory and orchestration around model calls

You can use `ftkAI` without PRM.

You can also use PRM with the local `ftkAI` model by wrapping generation calls, which is what `prm_generate.py` does.

## Current Integration Status

PRM is present and usable, but it is not the default runtime path for the main training/inference scripts.

Current practical status:

- `prm_generate.py` uses PRM directly
- base training in `train/train.py` does not automatically call PRM
- base inference in `generate.py` and `ask_model.py` does not automatically call PRM
- `PRM/integration.py` exists for custom wiring when you want PRM-aware model execution

## When To Use PRM

Use PRM when the task is multi-step and stateful, for example:

- long coding sessions
- project management or staged task execution
- workflows where the model should remember prior conclusions
- refinement loops where failed attempts should affect the next step

Do not use PRM just to train the base model on offline corpora. The normal data pipeline and training stack already handle that.

## Notes On Curriculum

`PRM/curriculum.py` is a curriculum utility module. It is separate from the main streaming-safe training path.

Important distinction:

- PRM curriculum logic is map-style and sampler-based
- the new Kaggle-safe streaming data path in `ftkAI` uses `IterableDataset`
- those two modes are intentionally not combined in the current training setup
