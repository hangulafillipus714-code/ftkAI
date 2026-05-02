"""
PRM Controller — Main Orchestration Loop

Implements the THINK → ACT → UPDATE cycle described in the design doc:

    Step 1  User gives task
    Step 2  Load PRM from store
    Step 3  Build prompt with PRM injected
    Step 4  Model produces output
    Step 5  Memory Updater applies delta
    Step 6  Store updated PRM

The controller is model-agnostic.  It accepts a callable `model_fn` that
takes a prompt string and returns a response string.  Plug in any backend:
OpenAI, local transformers, Anthropic, etc.

For your custom ModernLLM, wrap it:
    def model_fn(prompt: str) -> str:
        tokens = tokenizer(prompt)
        return tokenizer.decode(llm(tokens))

    controller = PRMController(model_fn=model_fn, store=store, project_id="...")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .compressor import CompressionConfig, PRMCompressor
from .injection import InjectionFormat, build_prompt_block
from .schema import PRMState
from .store import PRMStore
from .updater import PRMUpdater, UpdateDelta

logger = logging.getLogger(__name__)

ModelFn = Callable[[str], str]


# ─────────────────────────────────────────────
# Controller configuration
# ─────────────────────────────────────────────


@dataclass
class ControllerConfig:
    # Injection settings
    injection_format: InjectionFormat = InjectionFormat.TEXT
    include_bugs: bool = True
    include_memory: bool = True
    include_tasks: bool = True
    max_injection_chars: int = 6_000

    # Update settings
    update_mode: str = "auto"          # "auto" | "structured" | "heuristic"
    compress_every: int = 10

    # Retry on model errors
    max_retries: int = 2
    retry_delay: float = 1.0           # seconds

    # Whether to always save after every step
    auto_save: bool = True

    # Compression config (passed to PRMCompressor)
    compression: CompressionConfig = field(default_factory=CompressionConfig)


# ─────────────────────────────────────────────
# Turn result
# ─────────────────────────────────────────────


@dataclass
class TurnResult:
    prompt: str
    model_output: str
    updated_state: PRMState
    latency_ms: float
    compressed: bool
    error: Optional[str] = None


# ─────────────────────────────────────────────
# Controller
# ─────────────────────────────────────────────


class PRMController:
    """
    Stateful orchestrator for one project session.

    Usage:
        store = PRMStore("./prm.db")
        state = PRMState(project_name="my_llm", goal="Train a 1B param model")
        store.save(state)

        def my_model(prompt: str) -> str:
            ...  # your inference call

        ctrl = PRMController(
            model_fn=my_model,
            store=store,
            project_id=state.project_id,
        )

        result = ctrl.step("Write the training loop")
        print(result.model_output)
    """

    def __init__(
        self,
        model_fn: ModelFn,
        store: PRMStore,
        project_id: str,
        config: Optional[ControllerConfig] = None,
    ) -> None:
        self.model_fn = model_fn
        self.store = store
        self.project_id = project_id
        self.config = config or ControllerConfig()
        self._updater = PRMUpdater(
            compress_every=self.config.compress_every,
            compression_config=self.config.compression,
        )
        self._compressor = PRMCompressor(self.config.compression)
        self._history: List[TurnResult] = []

    # ── Main loop ────────────────────────────

    def step(self, user_input: str, extra_context: str = "") -> TurnResult:
        """Execute one THINK → ACT → UPDATE cycle."""
        result = self._execute_turn(user_input, extra_context)
        self._history.append(result)
        return result

    def step_with_refinement(
        self,
        user_input: str,
        extra_context: str = "",
        max_refinement_steps: int = 1,
        min_confidence: float = 0.6,
    ) -> TurnResult:
        """
        Execute a THINK → ACT → UPDATE cycle with potential refinement.
        If the model detects a bug or has low confidence, it is asked to re-think.
        """
        initial_state = self._load_state()
        if initial_state is None:
            raise RuntimeError(f"Project '{self.project_id}' not found in store")
            
        # Capture timestamp before execution
        pre_turn_timestamp = initial_state.updated_at
        
        result = self._execute_turn(user_input, extra_context)
        
        for i in range(max_refinement_steps):
            state = result.updated_state
            
            # Check for triggers: new bugs added in this specific turn
            has_new_bugs = any(
                not b.resolved and b.created_at > pre_turn_timestamp 
                for b in state.bugs
            )
            low_confidence = state.goal_confidence < min_confidence
            
            if not (has_new_bugs or low_confidence):
                break
                
            logger.info("Refinement triggered (step %d/%d). Reasons: bugs=%s, low_conf=%s", 
                        i+1, max_refinement_steps, has_new_bugs, low_confidence)
            
            refinement_prompt = (
                "Your previous response indicated high uncertainty or detected potential bugs. "
                "Please review the PRM state (especially the bugs list) and provide a "
                "corrected or refined solution that addresses those issues."
            )
            
            # Update pre-turn timestamp for next refinement loop
            pre_turn_timestamp = state.updated_at
            
            # Execute refinement turn
            result = self._execute_turn(refinement_prompt, extra_context=f"PREVIOUS_OUTPUT: {result.model_output}")
            
        self._history.append(result)
        return result

    def _execute_turn(self, user_input: str, extra_context: str = "") -> TurnResult:
        """Internal turn implementation."""
        t0 = time.monotonic()

        # ── STEP 2: Load PRM ────────────────
        state = self._load_state()
        if state is None:
            raise RuntimeError(f"Project '{self.project_id}' not found in store")

        # ── STEP 3: Build prompt ─────────────
        prm_block = build_prompt_block(
            state,
            fmt=self.config.injection_format,
            include_bugs=self.config.include_bugs,
            include_memory=self.config.include_memory,
            include_tasks=self.config.include_tasks,
            max_chars=self.config.max_injection_chars,
        )
        prompt = self._build_prompt(prm_block, user_input, extra_context)

        # ── STEP 4: Call model ───────────────
        output, error = self._call_model(prompt)

        # ── STEP 5: Update memory ────────────
        compressed = False
        prev_version = state.version
        if output:
            state = self._updater.apply(state, output, mode=self.config.update_mode)
            # check for compression by comparing runs
            # (dirty but works: we check if compression runs increased)
            # Actually, apply() doesn't return if it compressed easily, 
            # so we just trust the internal logic.

        # ── STEP 6: Save ─────────────────────
        if self.config.auto_save and output:
            self.store.save(state)
            logger.info(
                "Saved PRM v%d → v%d (project='%s')",
                prev_version,
                state.version,
                state.project_name,
            )

        elapsed_ms = (time.monotonic() - t0) * 1000
        return TurnResult(
            prompt=prompt,
            model_output=output or "",
            updated_state=state,
            latency_ms=round(elapsed_ms, 1),
            compressed=False, # will be detailed next
            error=error,
        )

    def apply_manual_delta(self, delta: UpdateDelta, force_compress: bool = False) -> PRMState:
        """
        Apply a manually constructed UpdateDelta without calling the model.
        Useful for seeding initial project state or applying tool results.
        """
        state = self._load_state()
        if state is None:
            raise RuntimeError(f"Project '{self.project_id}' not found in store")
        new_state = self._updater.apply_delta(state, delta, force_compress=force_compress)
        if self.config.auto_save:
            self.store.save(new_state)
        return new_state

    def force_compress(self) -> PRMState:
        """Manually trigger compression and save the result."""
        state = self._load_state()
        if state is None:
            raise RuntimeError(f"Project '{self.project_id}' not found in store")
        compressed = self._compressor.run(state)
        self.store.save(compressed)
        return compressed

    def status(self) -> Dict[str, Any]:
        """Return a human-readable status dict for the current project."""
        state = self._load_state()
        if state is None:
            return {"error": "project not found"}
        return {
            **state.summary_stats(),
            "project_name": state.project_name,
            "goal": state.goal,
            "stage": state.current_stage,
            "next_actions": state.next_actions(),
            "turns_this_session": len(self._history),
        }

    def history(self) -> List[TurnResult]:
        return list(self._history)

    # ── Private helpers ──────────────────────

    def _load_state(self) -> Optional[PRMState]:
        return self.store.load(self.project_id)

    def _build_prompt(self, prm_block: str, user_input: str, extra: str) -> str:
        parts = [prm_block, "## User Request", user_input]
        if extra:
            parts.append(f"## Context\n{extra}")
        parts.append(
            "\n## Instructions\n"
            "Read the project state above carefully.\n"
            "Produce your response.\n"
            "If you need to update the project state, append a ```json block with key "
            '"prm_update" containing only changed fields.\n'
            "Do NOT include raw conversation history in the update block."
        )
        return "\n\n".join(parts)

    def _call_model(self, prompt: str) -> tuple[Optional[str], Optional[str]]:
        last_err = None
        for attempt in range(self.config.max_retries + 1):
            try:
                output = self.model_fn(prompt)
                return output, None
            except Exception as exc:
                last_err = str(exc)
                logger.warning(
                    "Model call failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.config.max_retries + 1,
                    exc,
                )
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay)
        return None, last_err


# ─────────────────────────────────────────────
# Factory helpers
# ─────────────────────────────────────────────


def create_project(
    store: PRMStore,
    name: str,
    goal: str,
    constraints: Optional[List[str]] = None,
) -> PRMState:
    """Create and persist a new PRM project."""
    state = PRMState(
        project_name=name,
        goal=goal,
        constraints=constraints or [],
    )
    store.save(state)
    logger.info("Created project '%s' (id=%s)", name, state.project_id)
    return state


def load_or_create(
    store: PRMStore,
    name: str,
    goal: str = "",
    constraints: Optional[List[str]] = None,
) -> PRMState:
    """Load by name, or create if not found."""
    existing = store.load_by_name(name)
    if existing is not None:
        return existing
    return create_project(store, name, goal, constraints)
