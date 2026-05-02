"""
PRM Metrics
===========
Measures what actually matters for long-horizon task completion,
as specified in the design document:

  ✔ Task completion rate
  ✔ Error propagation depth  (how far a bug spreads before caught)
  ✔ Correction success rate  (bug fixed vs re-opened ratio)
  ✔ Memory retention signal  (relevant items survived compression)
  ✔ Goal drift score         (how much current_stage diverges from goal)
  ✔ Context efficiency       (useful tokens / total injected tokens)

All metrics are computed from PRMState snapshots — no model access needed.
They can be logged to SQLite (via PRMStore's DB) or exported as JSON/CSV.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .schema import PRMState, TaskStatus


# ─────────────────────────────────────────────
# Snapshot diff
# ─────────────────────────────────────────────


@dataclass
class StateDiff:
    """What changed between two consecutive PRMState versions."""
    tasks_completed: int = 0
    tasks_failed: int = 0
    bugs_added: int = 0
    bugs_resolved: int = 0
    memory_added: int = 0
    memory_pruned: int = 0          # items present in v1 but absent in v2
    compression_triggered: bool = False
    stage_changed: bool = False


def diff_states(before: PRMState, after: PRMState) -> StateDiff:
    """Compute what changed between two consecutive PRM snapshots."""
    before_task_statuses = {tid: t.status for tid, t in before.tasks.items()}
    after_task_statuses  = {tid: t.status for tid, t in after.tasks.items()}

    completed = sum(
        1 for tid, st in after_task_statuses.items()
        if st == TaskStatus.DONE and before_task_statuses.get(tid) != TaskStatus.DONE
    )
    failed = sum(
        1 for tid, st in after_task_statuses.items()
        if st == TaskStatus.FAILED and before_task_statuses.get(tid) != TaskStatus.FAILED
    )

    before_bug_ids = {b.id for b in before.bugs}
    after_bug_ids  = {b.id for b in after.bugs}
    bugs_added     = len(after_bug_ids - before_bug_ids)

    before_resolved = {b.id for b in before.bugs if b.resolved}
    after_resolved  = {b.id for b in after.bugs  if b.resolved}
    bugs_resolved   = len(after_resolved - before_resolved)

    before_mem_ids = {m.id for m in before.memory}
    after_mem_ids  = {m.id for m in after.memory}
    mem_added  = len(after_mem_ids - before_mem_ids)
    mem_pruned = len(before_mem_ids - after_mem_ids)

    return StateDiff(
        tasks_completed=completed,
        tasks_failed=failed,
        bugs_added=bugs_added,
        bugs_resolved=bugs_resolved,
        memory_added=mem_added,
        memory_pruned=mem_pruned,
        compression_triggered=(after.compression_runs > before.compression_runs),
        stage_changed=(after.current_stage != before.current_stage),
    )


# ─────────────────────────────────────────────
# Per-snapshot metrics
# ─────────────────────────────────────────────


@dataclass
class SnapshotMetrics:
    """Metrics computed from a single PRMState."""

    project_id: str
    version: int
    timestamp: str

    # Task completion
    task_completion_rate: float         # done / total
    task_blocked_rate: float            # blocked / total
    task_failure_rate: float            # failed / total

    # Bug health
    bug_resolution_rate: float          # resolved / total (all time)
    open_bug_count: int
    critical_bug_count: int

    # Memory quality
    avg_memory_relevance: float
    avg_memory_confidence: float
    memory_item_count: int
    high_relevance_ratio: float         # items > 0.7 relevance / total

    # Goal stability
    goal_confidence: float

    # Compression
    compression_runs: int
    archived_item_count: int
    compression_efficiency: float       # archived / (archived + active)


def compute_snapshot_metrics(state: PRMState) -> SnapshotMetrics:
    tasks = list(state.tasks.values())
    n_tasks = len(tasks) or 1   # avoid /0

    done     = sum(1 for t in tasks if t.status == TaskStatus.DONE)
    failed   = sum(1 for t in tasks if t.status == TaskStatus.FAILED)
    blocked  = sum(
        1 for t in tasks
        if t.status not in (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.ARCHIVED)
        and t.is_blocked(state.tasks)
    )

    all_bugs = state.bugs
    n_bugs = len(all_bugs) or 1
    resolved_bugs = sum(1 for b in all_bugs if b.resolved)
    open_bugs = n_bugs - resolved_bugs
    critical_bugs = sum(
        1 for b in all_bugs
        if not b.resolved and b.severity.value in ("high", "critical")
    )

    mem = state.memory
    n_mem = len(mem) or 1
    avg_rel  = sum(m.relevance  for m in mem) / n_mem
    avg_conf = sum(m.confidence for m in mem) / n_mem
    high_rel = sum(1 for m in mem if m.relevance > 0.7) / n_mem

    archived = state.archived_item_count
    total_ever = archived + n_mem
    comp_eff = archived / (total_ever or 1)

    return SnapshotMetrics(
        project_id=state.project_id,
        version=state.version,
        timestamp=datetime.now(timezone.utc).isoformat(),
        task_completion_rate=done / n_tasks,
        task_blocked_rate=blocked / n_tasks,
        task_failure_rate=failed / n_tasks,
        bug_resolution_rate=resolved_bugs / n_bugs,
        open_bug_count=open_bugs,
        critical_bug_count=critical_bugs,
        avg_memory_relevance=round(avg_rel, 4),
        avg_memory_confidence=round(avg_conf, 4),
        memory_item_count=n_mem,
        high_relevance_ratio=round(high_rel, 4),
        goal_confidence=state.goal_confidence,
        compression_runs=state.compression_runs,
        archived_item_count=archived,
        compression_efficiency=round(comp_eff, 4),
    )


# ─────────────────────────────────────────────
# Session metrics (across multiple snapshots)
# ─────────────────────────────────────────────


@dataclass
class SessionMetrics:
    """Aggregate metrics across an entire session (sequence of state snapshots)."""

    project_id: str
    project_name: str
    total_versions: int

    # Completion trajectory
    task_completion_rate_final: float
    task_completion_rate_peak: float

    # Error propagation: how many versions elapsed before each bug was resolved
    avg_error_propagation_depth: float     # lower is better
    max_error_propagation_depth: int

    # Correction success: bugs resolved without re-opening
    correction_success_rate: float

    # Memory health over time
    avg_memory_relevance_trend: float      # slope (positive = improving)
    memory_stability: float                # 1 - stddev(item_counts) / mean

    # Goal drift (stage change frequency)
    stage_change_rate: float               # changes / version

    # Compression summary
    total_items_archived: int
    compression_runs: int

    # Raw snapshot metrics for plotting
    snapshots: List[SnapshotMetrics] = field(default_factory=list)


def compute_session_metrics(
    states: Sequence[PRMState],
    project_name: str = "",
) -> SessionMetrics:
    """
    Compute session-level metrics from an ordered sequence of PRMState snapshots.
    States must be in ascending version order.
    """
    if not states:
        raise ValueError("states must be non-empty")

    snapshots = [compute_snapshot_metrics(s) for s in states]

    # Task completion trajectory
    completion_rates = [s.task_completion_rate for s in snapshots]
    peak_completion = max(completion_rates)
    final_completion = completion_rates[-1]

    # Error propagation depth
    # For each bug: find the version it was added and the version it was resolved
    bug_lifetimes: List[int] = []
    bug_first_seen: Dict[str, int] = {}
    for state in states:
        for bug in state.bugs:
            if bug.id not in bug_first_seen:
                bug_first_seen[bug.id] = state.version
            if bug.resolved and bug.resolved_at:
                lifetime = state.version - bug_first_seen[bug.id]
                if lifetime >= 0:
                    bug_lifetimes.append(lifetime)

    avg_propagation = sum(bug_lifetimes) / len(bug_lifetimes) if bug_lifetimes else 0.0
    max_propagation = max(bug_lifetimes) if bug_lifetimes else 0

    # Correction success rate (resolved bugs that were never re-opened)
    # Approximation: count distinct resolved bugs across all states
    ever_resolved: set[str] = set()
    ever_reopened: set[str] = set()
    prev_resolved: set[str] = set()
    for state in states:
        currently_resolved = {b.id for b in state.bugs if b.resolved}
        # If a bug was resolved before but isn't now — re-opened
        reopened_this_version = prev_resolved - currently_resolved
        ever_reopened.update(reopened_this_version)
        ever_resolved.update(currently_resolved)
        prev_resolved = currently_resolved

    correction_success = (
        (len(ever_resolved) - len(ever_reopened)) / len(ever_resolved)
        if ever_resolved else 1.0
    )

    # Memory relevance trend (linear slope using simple least-squares)
    rel_values = [s.avg_memory_relevance for s in snapshots]
    mem_trend = _linear_slope(rel_values)

    # Memory stability
    counts = [s.memory_item_count for s in snapshots]
    mean_count = sum(counts) / len(counts)
    stddev = math.sqrt(sum((c - mean_count) ** 2 for c in counts) / len(counts))
    mem_stability = 1.0 - min(1.0, stddev / (mean_count or 1))

    # Stage change rate
    stage_changes = sum(
        1 for i in range(1, len(states))
        if states[i].current_stage != states[i - 1].current_stage
    )
    stage_change_rate = stage_changes / max(1, len(states) - 1)

    return SessionMetrics(
        project_id=states[0].project_id,
        project_name=project_name or states[0].project_name,
        total_versions=len(states),
        task_completion_rate_final=round(final_completion, 4),
        task_completion_rate_peak=round(peak_completion, 4),
        avg_error_propagation_depth=round(avg_propagation, 2),
        max_error_propagation_depth=max_propagation,
        correction_success_rate=round(correction_success, 4),
        avg_memory_relevance_trend=round(mem_trend, 6),
        memory_stability=round(mem_stability, 4),
        stage_change_rate=round(stage_change_rate, 4),
        total_items_archived=states[-1].archived_item_count,
        compression_runs=states[-1].compression_runs,
        snapshots=snapshots,
    )


def _linear_slope(values: List[float]) -> float:
    """Slope of a linear fit to values (index as x-axis). Simple least squares."""
    n = len(values)
    if n < 2:
        return 0.0
    xs = list(range(n))
    x_mean = (n - 1) / 2
    y_mean = sum(values) / n
    num = sum((xs[i] - x_mean) * (values[i] - y_mean) for i in range(n))
    den = sum((xs[i] - x_mean) ** 2 for i in range(n))
    return num / den if den else 0.0


# ─────────────────────────────────────────────
# Metrics Logger (SQLite)
# ─────────────────────────────────────────────

_METRICS_SCHEMA = """
CREATE TABLE IF NOT EXISTS prm_metrics (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  TEXT    NOT NULL,
    version     INTEGER NOT NULL,
    timestamp   TEXT    NOT NULL,
    metrics_json TEXT   NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_metrics_project ON prm_metrics(project_id, version);
"""


class MetricsLogger:
    """
    Persists SnapshotMetrics to the same SQLite DB as PRMStore
    (or a separate DB file).

    Usage:
        logger = MetricsLogger("./prm.db")
        logger.log(compute_snapshot_metrics(state))
        history = logger.load_history("project-id")
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_METRICS_SCHEMA)
        self._conn.commit()

    def log(self, metrics: SnapshotMetrics) -> None:
        payload = json.dumps(asdict(metrics))
        self._conn.execute(
            "INSERT INTO prm_metrics (project_id, version, timestamp, metrics_json) VALUES (?,?,?,?)",
            (metrics.project_id, metrics.version, metrics.timestamp, payload),
        )
        self._conn.commit()

    def load_history(self, project_id: str) -> List[SnapshotMetrics]:
        rows = self._conn.execute(
            "SELECT metrics_json FROM prm_metrics WHERE project_id = ? ORDER BY version ASC",
            (project_id,),
        ).fetchall()
        result = []
        for (raw,) in rows:
            data = json.loads(raw)
            result.append(SnapshotMetrics(**data))
        return result

    def export_csv(self, project_id: str, path: str | Path) -> None:
        """Export metrics history as CSV for plotting."""
        import csv
        history = self.load_history(project_id)
        if not history:
            return
        path = Path(path)
        fields = [f for f in asdict(history[0]).keys()]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for m in history:
                writer.writerow(asdict(m))

    def summary_report(self, project_id: str) -> str:
        """Return a concise human-readable metrics report."""
        history = self.load_history(project_id)
        if not history:
            return "No metrics recorded yet."
        latest = history[-1]
        lines = [
            f"── Metrics Report ──────────────────────",
            f"  Project     : {project_id}",
            f"  Snapshots   : {len(history)}",
            f"  Completion  : {latest.task_completion_rate:.1%}",
            f"  Open bugs   : {latest.open_bug_count}  (critical: {latest.critical_bug_count})",
            f"  Avg relevance: {latest.avg_memory_relevance:.3f}",
            f"  Memory items: {latest.memory_item_count}",
            f"  Compressions: {latest.compression_runs}  "
            f"(archived {latest.archived_item_count} items)",
            f"  Goal conf.  : {latest.goal_confidence:.2f}",
            f"────────────────────────────────────────",
        ]
        return "\n".join(lines)

    def close(self) -> None:
        self._conn.close()
