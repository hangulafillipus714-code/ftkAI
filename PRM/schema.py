"""
PRM Schema — Persistent Reasoning Memory
Core data models for structured project state.

Design principles:
  - Every field is typed and validated via Pydantic
  - Compression-friendly: every item carries relevance + confidence
  - No raw logs, no uncompressed history
  - JSON-serializable for storage and injection
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    DONE = "done"
    FAILED = "failed"
    ARCHIVED = "archived"


class MemoryItemKind(str, Enum):
    DECISION = "decision"
    BUG = "bug"
    CONSTRAINT = "constraint"
    INSIGHT = "insight"
    MODULE = "module"
    DEPENDENCY = "dependency"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ─────────────────────────────────────────────
# Leaf-level models
# ─────────────────────────────────────────────


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uid() -> str:
    return str(uuid.uuid4())[:8]


class ScoredItem(BaseModel):
    """
    Any piece of information stored in PRM.
    Carries its own relevance and confidence so the compressor can prune it.
    """

    id: str = Field(default_factory=_uid)
    kind: MemoryItemKind
    content: str = Field(..., min_length=1, max_length=4096)
    relevance: float = Field(default=1.0, ge=0.0, le=1.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    status: TaskStatus = TaskStatus.IN_PROGRESS
    tags: List[str] = Field(default_factory=list)
    locked: bool = Field(default=False)
    access_count: int = Field(default=0, ge=0)
    last_used: Optional[str] = None
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("tags")
    @classmethod
    def deduplicate_tags(cls, v: List[str]) -> List[str]:
        return list(dict.fromkeys(v))

    def touch(self) -> None:
        self.updated_at = _now()

    def record_access(self) -> None:
        self.access_count += 1
        self.last_used = _now()

    def decay(self, factor: float = 0.95) -> None:
        """Slightly reduce relevance over time — keeps hot items alive."""
        self.relevance = round(max(0.0, self.relevance * factor), 4)


class TaskNode(BaseModel):
    """
    A node in the project task graph.
    Tasks can depend on each other; cycles are forbidden.
    """

    id: str = Field(default_factory=_uid)
    title: str = Field(..., min_length=1, max_length=256)
    description: str = Field(default="")
    status: TaskStatus = TaskStatus.PENDING
    priority: Priority = Priority.MEDIUM
    depends_on: List[str] = Field(default_factory=list)   # IDs of blocking tasks
    next_action: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)

    def touch(self) -> None:
        self.updated_at = _now()

    def is_blocked(self, all_tasks: Dict[str, "TaskNode"]) -> bool:
        return any(
            all_tasks[dep_id].status not in (TaskStatus.DONE, TaskStatus.ARCHIVED)
            for dep_id in self.depends_on
            if dep_id in all_tasks
        )


class BugReport(BaseModel):
    """Tracked bug — never let the model repeat a known mistake."""

    id: str = Field(default_factory=_uid)
    summary: str = Field(..., min_length=1, max_length=1024)
    location: Optional[str] = None   # e.g. "module.py:forward()"
    severity: Priority = Priority.MEDIUM
    resolved: bool = False
    fix_description: Optional[str] = None
    created_at: str = Field(default_factory=_now)
    resolved_at: Optional[str] = None

    def resolve(self, fix: str) -> None:
        self.resolved = True
        self.fix_description = fix
        self.resolved_at = _now()


# ─────────────────────────────────────────────
# Top-level PRM State
# ─────────────────────────────────────────────


class PRMState(BaseModel):
    """
    The complete Persistent Reasoning Memory state for one project/session.

    This is the single source of truth.  Never store raw conversation logs
    here — only compressed, structured intelligence.
    """

    # Identity
    project_id: str = Field(default_factory=_uid)
    project_name: str = Field(default="unnamed_project")
    version: int = Field(default=1, ge=1)
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)

    # Core cognition fields
    goal: str = Field(default="", max_length=2048)
    current_stage: str = Field(default="")
    constraints: List[str] = Field(default_factory=list)

    # Task graph
    tasks: Dict[str, TaskNode] = Field(default_factory=dict)

    # Scored memory items
    memory: List[ScoredItem] = Field(default_factory=list)

    # Bug tracker
    bugs: List[BugReport] = Field(default_factory=list)

    # Compression bookkeeping
    total_updates: int = Field(default=0, ge=0)
    compression_runs: int = Field(default=0, ge=0)
    archived_item_count: int = Field(default=0, ge=0)

    # Free-form confidence on overall project direction
    goal_confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _no_cycle_check(self) -> "PRMState":
        """Light cycle detection on task dependency graph."""
        visited: set[str] = set()

        def visit(tid: str, stack: set[str]) -> None:
            if tid in stack:
                raise ValueError(f"Cycle detected in task dependency graph at task '{tid}'")
            if tid in visited:
                return
            stack.add(tid)
            task = self.tasks.get(tid)
            for dep in (task.depends_on if task else []):
                visit(dep, stack)
            stack.discard(tid)
            visited.add(tid)

        for tid in list(self.tasks.keys()):
            visit(tid, set())
        return self

    # ── Convenience helpers ──────────────────

    def touch(self) -> None:
        self.updated_at = _now()
        self.total_updates += 1
        self.version += 1

    def active_tasks(self) -> List[TaskNode]:
        return [t for t in self.tasks.values() if t.status not in (TaskStatus.DONE, TaskStatus.ARCHIVED)]

    def unresolved_bugs(self) -> List[BugReport]:
        return [b for b in self.bugs if not b.resolved]

    def next_actions(self) -> List[str]:
        """Collect next_action strings from all non-done tasks."""
        actions = []
        for t in self.tasks.values():
            if t.status not in (TaskStatus.DONE, TaskStatus.ARCHIVED) and t.next_action:
                actions.append(f"[{t.priority.upper()}] {t.title}: {t.next_action}")
        return actions

    def memory_by_kind(self, kind: MemoryItemKind) -> List[ScoredItem]:
        return [m for m in self.memory if m.kind == kind]

    def add_memory_item(self, item: ScoredItem) -> None:
        self.memory.append(item)
        self.touch()

    def add_task(self, task: TaskNode) -> None:
        self.tasks[task.id] = task
        self.touch()

    def add_bug(self, bug: BugReport) -> None:
        self.bugs.append(bug)
        self.touch()

    def summary_stats(self) -> Dict[str, Any]:
        return {
            "total_tasks": len(self.tasks),
            "active_tasks": len(self.active_tasks()),
            "done_tasks": sum(1 for t in self.tasks.values() if t.status == TaskStatus.DONE),
            "total_memory_items": len(self.memory),
            "unresolved_bugs": len(self.unresolved_bugs()),
            "version": self.version,
            "compression_runs": self.compression_runs,
            "archived_items": self.archived_item_count,
        }
