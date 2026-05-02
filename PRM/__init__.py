"""
PRM — Persistent Reasoning Memory
==================================
A production-grade external memory system for language model agents.

Quick start:
    from prm import PRMStore, PRMController, PRMState, create_project

    store = PRMStore("./prm.db")
    state = create_project(store, "my_llm", goal="Train a custom 1B model")

    ctrl = PRMController(
        model_fn=lambda prompt: my_model.generate(prompt),
        store=store,
        project_id=state.project_id,
    )

    result = ctrl.step("Write the training loop with gradient clipping")
    print(result.model_output)
"""

from .compressor import CompressionConfig, PRMCompressor, score_new_item
from .controller import (
    ControllerConfig,
    PRMController,
    TurnResult,
    create_project,
    load_or_create,
)
from .injection import InjectionFormat, build_prompt_block
from .schema import (
    BugReport,
    MemoryItemKind,
    PRMState,
    Priority,
    ScoredItem,
    TaskNode,
    TaskStatus,
)
from .store import PRMStore
from .updater import PRMUpdater, UpdateDelta

__all__ = [
    # Schema
    "PRMState",
    "TaskNode",
    "TaskStatus",
    "BugReport",
    "ScoredItem",
    "MemoryItemKind",
    "Priority",
    # Store
    "PRMStore",
    # Controller
    "PRMController",
    "ControllerConfig",
    "TurnResult",
    "create_project",
    "load_or_create",
    # Updater
    "PRMUpdater",
    "UpdateDelta",
    # Injection
    "build_prompt_block",
    "InjectionFormat",
    # Compressor
    "PRMCompressor",
    "CompressionConfig",
    "score_new_item",
]

__version__ = "1.0.0"
