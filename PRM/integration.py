"""
ModernLLM ↔ PRM Integration
============================
Shows exactly how to wire PRM into your existing transformer architecture
WITHOUT touching the model weights or training loop.

Three integration levels:
  A) Prompt injection (zero model changes, production-ready today)
  B) Forward-pass PRM token injection (requires tokenizer change)

Level A is the recommended starting point.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, Optional

# ─────────────────────────────────────────────
# Level A — Prompt Injection (no model changes)
# ─────────────────────────────────────────────
#
# This is the recommended path.  The PRM state is serialised as a text block
# and prepended to every prompt.  The model reads it as context and (optionally)
# appends a structured update block to its response.
#
# ✔ Works with any model (transformers, API, local)
# ✔ No gradient changes
# ✔ Can be enabled/disabled per-call
# ─────────────────────────────────────────────

from prm import (
    InjectionFormat,
    PRMController,
    PRMState,
    PRMStore,
    build_prompt_block,
    create_project,
    load_or_create,
)

logger = logging.getLogger(__name__)


class PRMWrappedLLM:
    """
    Wraps any callable LLM with PRM memory injection.

    Usage:
        llm = PRMWrappedLLM(
            base_model_fn=my_model.generate,
            store=PRMStore("./prm.db"),
            project_id=state.project_id,
        )
        response = llm("Write the forward pass for grouped query attention")
    """

    def __init__(
        self,
        base_model_fn: Callable[[str], str],
        store: PRMStore,
        project_id: str,
        injection_format: InjectionFormat = InjectionFormat.TEXT,
    ) -> None:
        self._controller = PRMController(
            model_fn=base_model_fn,
            store=store,
            project_id=project_id,
        )
        self._controller.config.injection_format = injection_format

    def __call__(self, user_request: str, extra_context: str = "") -> str:
        result = self._controller.step(user_request, extra_context=extra_context)
        if result.error:
            logger.error("Model error: %s", result.error)
        return result.model_output

    @property
    def state(self) -> Optional[PRMState]:
        return self._controller.store.load(self._controller.project_id)

    def status(self) -> Dict[str, Any]:
        return self._controller.status()


# ─────────────────────────────────────────────
# Level B — Forward-pass PRM token injection
# ─────────────────────────────────────────────
#
# For architectures where you control tokenisation and the forward pass.
# Inject PRM as a prefix token sequence before the user tokens.
#
# No weight changes required.  This is pure prompt engineering at the
# token level — you can cache the PRM prefix KV to save compute.
# ─────────────────────────────────────────────


def inject_prm_tokens(
    model_forward: Callable,        # your model's forward(tokens, **kwargs) function
    tokenizer: Any,                 # your tokenizer (needs encode/decode)
    state: PRMState,
    user_tokens: Any,               # already-encoded user prompt tokens
    fmt: InjectionFormat = InjectionFormat.TEXT,
) -> Any:
    """
    Prepend PRM state tokens to user tokens before calling model.forward().

    Example (pseudocode):
        tokens = tokenizer.encode(user_prompt)
        output_tokens = inject_prm_tokens(model.forward, tokenizer, state, tokens)
        response = tokenizer.decode(output_tokens)
    """
    prm_text = build_prompt_block(state, fmt=fmt)
    prm_tokens = tokenizer.encode(prm_text)
    combined = _concat_tokens(prm_tokens, user_tokens)
    return model_forward(combined)


def _concat_tokens(a: Any, b: Any) -> Any:
    """Token concatenation — handles both Python lists and torch tensors."""
    try:
        import torch
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            return torch.cat([a, b], dim=-1)
    except ImportError:
        pass
    # Fallback: plain list concat
    return list(a) + list(b)



# ─────────────────────────────────────────────
# Example: full session using Level A
# ─────────────────────────────────────────────


def run_example_session() -> None:
    """
    Demonstrates a complete THINK → ACT → UPDATE cycle.
    Uses an echo model — replace with your actual model_fn.
    """
    store = PRMStore("./example_prm.db")
    state = load_or_create(
        store,
        name="custom_llm_project",
        goal="Implement and train a custom 1B-parameter transformer with GQA and MoE",
        constraints=[
            "Must run on 2× A100 80GB GPUs",
            "Max sequence length 8192 tokens",
            "Python 3.11 + PyTorch 2.3",
        ],
    )

    def echo_model(prompt: str) -> str:
        """Replace this with your real model call."""
        return (
            '```json\n{"prm_update": {"stage": "data pipeline", '
            '"new_memory": [{"kind": "decision", "content": '
            '"Using streaming DataLoader to avoid OOM on 8192 seq len", '
            '"confidence": 0.9}], '
            '"new_bugs": [{"summary": "DataLoader workers deadlock with >4 workers", '
            '"severity": "high", "location": "data/loader.py"}]}}\n```'
        )

    ctrl = PRMController(
        model_fn=echo_model,
        store=store,
        project_id=state.project_id,
    )

    # Seed initial tasks
    from prm import UpdateDelta
    from prm.schema import TaskNode, Priority
    delta = UpdateDelta(
        new_tasks=[
            {"title": "Data pipeline", "priority": "high", "next_action": "Implement streaming loader"},
            {"title": "Model architecture", "priority": "high", "next_action": "Wire GQA + MoE"},
            {"title": "Training loop", "priority": "medium", "next_action": "Add grad clipping"},
        ]
    )
    ctrl.apply_manual_delta(delta)

    result = ctrl.step("Build the streaming data pipeline for 8192 token sequences")
    print("=== Model Output ===")
    print(result.model_output)
    print("\n=== PRM Status ===")
    import json as _json
    print(_json.dumps(ctrl.status(), indent=2))

    store.export_json(state.project_id, "./example_prm_export.json")
    store.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_example_session()
