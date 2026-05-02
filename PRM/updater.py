"""
PRM Updater — Model Output → State Delta → Updated PRMState

This is the "Memory Updater" described in the design document.
It runs AFTER each model response, outside the transformer, and is
the sole mechanism through which the PRM state evolves.

Three update modes:
  1. STRUCTURED  — model output is a JSON dict conforming to UpdateDelta
  2. HEURISTIC   — plain text is parsed with lightweight heuristics
  3. PASSTHROUGH — caller supplies a pre-built UpdateDelta directly

The updater never stores raw text.  It extracts only the compressed,
structured intelligence that deserves a place in long-term memory.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .compressor import CompressionConfig, PRMCompressor, score_new_item
from .schema import (
    BugReport,
    MemoryItemKind,
    PRMState,
    Priority,
    ScoredItem,
    TaskNode,
    TaskStatus,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Update Delta — what the model is allowed to express
# ─────────────────────────────────────────────


@dataclass
class TaskUpdate:
    task_id: str
    status: Optional[str] = None          # TaskStatus value
    next_action: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class UpdateDelta:
    """
    Structured diff that the Memory Updater applies to PRMState.
    All fields are optional — partial updates are fine.
    """

    # High-level progress
    stage: Optional[str] = None
    goal_confidence: Optional[float] = None

    # New memory items to add
    new_memory: List[Dict[str, Any]] = field(default_factory=list)

    # Bugs to add
    new_bugs: List[Dict[str, Any]] = field(default_factory=list)

    # Bugs to resolve (by id)
    resolved_bug_ids: List[str] = field(default_factory=list)
    resolved_bug_fixes: Dict[str, str] = field(default_factory=dict)

    # New or updated tasks
    new_tasks: List[Dict[str, Any]] = field(default_factory=list)
    task_updates: List[TaskUpdate] = field(default_factory=list)

    # Constraints to add
    new_constraints: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────
# Updater
# ─────────────────────────────────────────────

# How many updates before we auto-compress
_COMPRESS_EVERY_N = 10


class PRMUpdater:
    """
    Takes model output (string or pre-parsed dict) and applies it to a PRMState.

    Typical call:
        updater = PRMUpdater()
        new_state = updater.apply(current_state, model_output_text)
    """

    def __init__(
        self,
        compress_every: int = _COMPRESS_EVERY_N,
        compression_config: Optional[CompressionConfig] = None,
    ) -> None:
        self.compress_every = compress_every
        self._compressor = PRMCompressor(compression_config)

    # ── Public entry point ───────────────────

    def apply(
        self,
        state: PRMState,
        model_output: str,
        mode: str = "auto",          # "auto" | "structured" | "heuristic"
        force_compress: bool = False,
    ) -> PRMState:
        """
        Parse model_output and return an updated (and possibly compressed) PRMState.

        Args:
            state:          Current PRM state.
            model_output:   Raw string from the model.
            mode:           Parsing strategy.
            force_compress: Always run compression after update.

        Returns:
            Updated PRMState (new object; state is not mutated).
        """
        delta = self._parse(model_output, mode)
        new_state = self._merge(state, delta)

        # Auto-compress every N updates
        should_compress = force_compress or (new_state.total_updates % self.compress_every == 0)
        if should_compress:
            new_state = self._compressor.run(new_state)

        return new_state

    def apply_delta(
        self,
        state: PRMState,
        delta: UpdateDelta,
        force_compress: bool = False,
    ) -> PRMState:
        """Apply a pre-built UpdateDelta directly (PASSTHROUGH mode)."""
        new_state = self._merge(state, delta)
        if force_compress:
            new_state = self._compressor.run(new_state)
        return new_state

    # ── Parsing ──────────────────────────────

    def _parse(self, text: str, mode: str) -> UpdateDelta:
        if mode == "structured":
            return self._parse_structured(text)
        if mode == "heuristic":
            return self._parse_heuristic(text)
        # auto: try structured first, fall back to heuristic
        try:
            delta = self._parse_structured(text)
            logger.debug("UpdateDelta parsed in STRUCTURED mode")
            return delta
        except (json.JSONDecodeError, KeyError, ValueError):
            logger.debug("Structured parse failed — falling back to heuristic")
            return self._parse_heuristic(text)

    @staticmethod
    def _parse_structured(text: str) -> UpdateDelta:
        """
        Expects a JSON block in the model output, optionally wrapped in ```json fences.
        The JSON must have a top-level "prm_update" key.
        """
        # Extract ```json ... ``` block if present
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        raw = fence_match.group(1) if fence_match else text

        # Find the first {...} block
        brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not brace_match:
            raise ValueError("No JSON object found")

        data: dict = json.loads(brace_match.group())
        update = data.get("prm_update", data)   # tolerate both wrapped and unwrapped

        delta = UpdateDelta()
        delta.stage = update.get("stage")
        delta.goal_confidence = update.get("goal_confidence")
        delta.new_constraints = update.get("new_constraints", [])
        delta.resolved_bug_ids = update.get("resolved_bug_ids", [])
        delta.resolved_bug_fixes = update.get("resolved_bug_fixes", {})

        for m in update.get("new_memory", []):
            delta.new_memory.append(m)
        for b in update.get("new_bugs", []):
            delta.new_bugs.append(b)
        for t in update.get("new_tasks", []):
            delta.new_tasks.append(t)
        for tu in update.get("task_updates", []):
            delta.task_updates.append(
                TaskUpdate(
                    task_id=tu["task_id"],
                    status=tu.get("status"),
                    next_action=tu.get("next_action"),
                    confidence=tu.get("confidence"),
                )
            )
        return delta

    @staticmethod
    def _parse_heuristic(text: str) -> UpdateDelta:
        """
        Extract signals from plain-text model output using lightweight regexes.
        This is intentionally conservative — better to miss something than to
        store noise in the PRM.
        """
        delta = UpdateDelta()
        lower = text.lower()

        # Stage detection
        stage_match = re.search(r"(?:stage|step|phase)[:\s]+([^\n.]{3,60})", lower)
        if stage_match:
            delta.stage = stage_match.group(1).strip()

        # Bug detection: "bug:", "issue:", "error:", "problem:", "fix:"
        bug_patterns = [
            r"(?:bug|issue|error|problem)[:\s]+([^\n.]{5,200})",
            r"(?:fixed?|resolved?)[:\s]+([^\n.]{5,200})",
        ]
        for pat in bug_patterns:
            for m in re.finditer(pat, lower):
                summary = m.group(1).strip()
                if len(summary) > 10:
                    delta.new_bugs.append({"summary": summary, "severity": "medium"})
                    break   # one bug per pattern is enough from heuristic

        # Decision detection: "decided", "chose", "approach:"
        decision_patterns = [r"(?:decided?|chose|approach)[:\s]+([^\n.]{10,300})"]
        for pat in decision_patterns:
            for m in re.finditer(pat, lower):
                content = m.group(1).strip()
                if len(content) > 15:
                    delta.new_memory.append(
                        {"kind": "decision", "content": content, "confidence": 0.75}
                    )
                    break

        # Constraint detection
        constraint_patterns = [r"(?:constraint|requirement|must|cannot)[:\s]+([^\n.]{10,200})"]
        for pat in constraint_patterns:
            for m in re.finditer(pat, lower):
                delta.new_constraints.append(m.group(1).strip())
                break

        return delta

    # ── Merging ──────────────────────────────

    @staticmethod
    def _merge(state: PRMState, delta: UpdateDelta) -> PRMState:
        """Apply delta to state, returning a new PRMState."""
        # Deep copy via Pydantic
        new = PRMState.model_validate_json(state.model_dump_json())

        # Stage / confidence
        if delta.stage:
            new.current_stage = delta.stage
        if delta.goal_confidence is not None:
            new.goal_confidence = max(0.0, min(1.0, delta.goal_confidence))

        # Constraints (deduplicate)
        existing_lower = {c.lower() for c in new.constraints}
        for c in delta.new_constraints:
            if c.lower() not in existing_lower:
                new.constraints.append(c)
                existing_lower.add(c.lower())

        # Memory items
        for m in delta.new_memory:
            kind = _parse_kind(m.get("kind", "insight"))
            relevance = m.get("relevance") or score_new_item(
                m.get("content", ""), kind, _parse_priority(m.get("priority", "medium"))
            )
            item = ScoredItem(
                kind=kind,
                content=str(m.get("content", ""))[:4096],
                relevance=float(relevance),
                confidence=float(m.get("confidence", 0.85)),
                tags=m.get("tags", []),
            )
            new.memory.append(item)

        # Bugs — add new
        for b in delta.new_bugs:
            bug = BugReport(
                summary=str(b.get("summary", ""))[:1024],
                location=b.get("location"),
                severity=_parse_priority(b.get("severity", "medium")),
            )
            new.bugs.append(bug)

        # Bugs — resolve
        resolved_map = {
            rid: delta.resolved_bug_fixes.get(rid, "resolved")
            for rid in delta.resolved_bug_ids
        }
        for bug in new.bugs:
            if bug.id in resolved_map and not bug.resolved:
                bug.resolve(resolved_map[bug.id])

        # New tasks
        for t in delta.new_tasks:
            task = TaskNode(
                title=str(t.get("title", "Unnamed Task"))[:256],
                description=t.get("description", ""),
                priority=_parse_priority(t.get("priority", "medium")),
                depends_on=t.get("depends_on", []),
                next_action=t.get("next_action"),
            )
            new.tasks[task.id] = task

        # Task updates
        for tu in delta.task_updates:
            if tu.task_id in new.tasks:
                task = new.tasks[tu.task_id]
                if tu.status:
                    try:
                        task.status = TaskStatus(tu.status)
                    except ValueError:
                        logger.warning("Unknown task status %r — ignored", tu.status)
                if tu.next_action is not None:
                    task.next_action = tu.next_action
                if tu.confidence is not None:
                    task.confidence = max(0.0, min(1.0, tu.confidence))
                task.touch()
            else:
                logger.warning("Task update references unknown task_id %r — ignored", tu.task_id)

        new.touch()
        return new


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────


def _parse_kind(raw: str) -> MemoryItemKind:
    mapping = {k.value: k for k in MemoryItemKind}
    return mapping.get(raw.lower().strip(), MemoryItemKind.INSIGHT)


def _parse_priority(raw: str) -> Priority:
    mapping = {p.value: p for p in Priority}
    return mapping.get(raw.lower().strip(), Priority.MEDIUM)
