"""
PRM Compressor
Keeps memory lean and signal-rich by:
  - Decaying relevance of old items
  - Archiving completed tasks below a threshold
  - Merging near-duplicate memory items (fuzzy dedup)
  - Summarising dense memory blocks into single items
  - Never letting the PRM grow into a raw log

Configuration is via CompressionConfig — all thresholds are explicit.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

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
# Configuration
# ─────────────────────────────────────────────


@dataclass
class CompressionConfig:
    # Items with relevance below this are archived / dropped
    prune_relevance_threshold: float = 0.15
    # Items marked DONE with relevance below this are archived
    done_relevance_threshold: float = 0.40
    # Decay factor applied to ALL items on each compression run
    decay_factor: float = 0.92
    # Tasks in DONE/ARCHIVED state older than N versions are archived
    task_archive_age: int = 5
    # Maximum memory items before forced prune (oldest/lowest relevance first)
    max_memory_items: int = 80
    # Maximum bug reports to keep in active list (resolved ones get archived faster)
    max_resolved_bugs: int = 10
    # Boost relevance for items matching these high-value kinds
    boost_kinds: Set[MemoryItemKind] = field(
        default_factory=lambda: {MemoryItemKind.BUG, MemoryItemKind.CONSTRAINT}
    )
    boost_amount: float = 0.05
    # Density Summarization threshold (items of same kind/tags)
    cluster_merge_threshold: int = 3
    # Resistance to decay per access count
    access_resistance_factor: float = 0.01


# ─────────────────────────────────────────────
# Compressor
# ─────────────────────────────────────────────


class PRMCompressor:
    """
    Stateless compressor — takes a PRMState and returns a compressed copy.
    The original state is never mutated; caller decides whether to adopt result.
    """

    def __init__(self, config: Optional[CompressionConfig] = None) -> None:
        self.config = config or CompressionConfig()

    def run(self, state: PRMState) -> PRMState:
        """
        Full compression pipeline.  Returns a new PRMState with:
          - decayed relevance scores
          - pruned low-relevance items
          - archived done tasks
          - deduplicated memory
          - summarized dense clusters
          - resolved bugs rotated out

        The state's compression_runs counter and archived_item_count are updated.
        """
        cfg = self.config

        # Work on a deep copy via Pydantic serialisation
        working = PRMState.model_validate_json(state.model_dump_json())
        archived_count = 0

        current_stage = working.current_stage.lower()

        # 1. Decay all memory items
        for item in working.memory:
            if item.locked:
                continue
                
            effective_decay = min(1.0, cfg.decay_factor + (item.access_count * cfg.access_resistance_factor))
            item.decay(effective_decay)
            
            # Boost important kinds to prevent them from fading out
            if item.kind in cfg.boost_kinds:
                item.relevance = round(min(1.0, item.relevance + cfg.boost_amount), 4)

            # Phase awareness boosts
            if "debug" in current_stage or "test" in current_stage:
                if item.kind == MemoryItemKind.BUG:
                    item.relevance = round(min(1.0, item.relevance + cfg.boost_amount * 2), 4)
            elif "arch" in current_stage or "design" in current_stage:
                if item.kind in (MemoryItemKind.DECISION, MemoryItemKind.CONSTRAINT):
                    item.relevance = round(min(1.0, item.relevance + cfg.boost_amount * 2), 4)

        # Build a single corpus string of active tasks to check for item ID references
        active_tasks_corpus = " ".join(
            f"{t.title} {t.description} {t.next_action or ''}" 
            for t in working.active_tasks()
        )

        # 2. Archive / prune memory items
        survivors: List[ScoredItem] = []
        for item in working.memory:
            if item.locked:
                survivors.append(item)
                continue

            is_referenced = item.id in active_tasks_corpus

            if item.status in (TaskStatus.DONE, TaskStatus.ARCHIVED):
                if item.relevance < cfg.done_relevance_threshold and not is_referenced:
                    archived_count += 1
                    continue   # drop from active memory
            if item.relevance < cfg.prune_relevance_threshold and not is_referenced:
                archived_count += 1
                continue
            survivors.append(item)

        # 3. Deduplicate memory items by fuzzy content fingerprint
        survivors = self._dedup_memory(survivors)
        
        # 4. Density Summarization of clustered memory variations
        survivors = self._summarize_clusters(survivors, cfg)

        # 5. Hard cap — if still too many, drop lowest relevance
        if len(survivors) > cfg.max_memory_items:
            survivors.sort(key=lambda i: i.relevance, reverse=True)
            # Find the first non-locked item to drop if we exceed limit.
            # Locked items cannot be dropped.
            unlocked = [i for i in survivors if not i.locked]
            locked = [i for i in survivors if i.locked]
            
            allowable_unlocked_size = max(0, cfg.max_memory_items - len(locked))
            dropped = max(0, len(unlocked) - allowable_unlocked_size)
            archived_count += dropped
            
            survivors = locked + unlocked[:allowable_unlocked_size]

        working.memory = survivors

        # 6. Archive completed tasks with low confidence or old age
        working.tasks = self._archive_tasks(working.tasks, cfg)

        # 7. Rotate resolved bugs
        working.bugs = self._rotate_bugs(working.bugs, cfg)

        # 8. Update bookkeeping
        working.compression_runs += 1
        working.archived_item_count += archived_count
        working.touch()

        logger.info(
            "Compression run #%d: archived %d items, memory %d→%d",
            working.compression_runs,
            archived_count,
            len(state.memory),
            len(working.memory),
        )
        return working

    # ── Private helpers ──────────────────────

    @staticmethod
    def _dedup_memory(items: List[ScoredItem]) -> List[ScoredItem]:
        """Remove items whose content is fuzzily identical (keep highest relevance copy)."""
        seen: Dict[str, ScoredItem] = {}
        for item in items:
            key = re.sub(r'[^a-z0-9]', '', item.content.lower())
            if not key:
                key = item.content.strip().lower()
            if key not in seen or item.relevance > seen[key].relevance:
                seen[key] = item
        return list(seen.values())

    @staticmethod
    def _summarize_clusters(items: List[ScoredItem], cfg: CompressionConfig) -> List[ScoredItem]:
        """Merge dense clusters of similar memory types into a single summary block."""
        clusters: Dict[tuple, List[ScoredItem]] = defaultdict(list)
        survivors = []
        
        for item in items:
            if item.locked:
                survivors.append(item)
            else:
                key = (item.kind, tuple(sorted(item.tags)))
                clusters[key].append(item)
                
        for key, cluster in clusters.items():
            if len(cluster) >= cfg.cluster_merge_threshold:
                kind, tags_t = key
                distilled_content = f"Dense cluster summary ({len(cluster)} items):\n"
                for idx, itm in enumerate(cluster):
                    short = itm.content if len(itm.content) < 200 else itm.content[:197] + "..."
                    distilled_content += f"- {short}\n"
                    
                summary_item = ScoredItem(
                    kind=kind,
                    content=distilled_content.strip()[:4096],
                    relevance=round(max(i.relevance for i in cluster), 4),
                    confidence=round(sum(i.confidence for i in cluster) / len(cluster), 4),
                    tags=list(tags_t),
                    access_count=sum(i.access_count for i in cluster)
                )
                survivors.append(summary_item)
            else:
                survivors.extend(cluster)
                
        return survivors

    @staticmethod
    def _archive_tasks(tasks: Dict[str, TaskNode], cfg: CompressionConfig) -> Dict[str, TaskNode]:
        """
        Move DONE tasks with low confidence to ARCHIVED status.
        (We keep them in the dict so dependency references stay valid.)
        """
        result: Dict[str, TaskNode] = {}
        for tid, task in tasks.items():
            if task.status == TaskStatus.DONE and task.confidence < 0.5:
                task.status = TaskStatus.ARCHIVED
                logger.debug("Archived low-confidence task '%s'", task.title)
            result[tid] = task
        return result

    @staticmethod
    def _rotate_bugs(bugs: List[BugReport], cfg: CompressionConfig) -> List[BugReport]:
        """
        Keep all unresolved bugs.  Trim resolved bug list to max_resolved_bugs.
        """
        unresolved = [b for b in bugs if not b.resolved]
        resolved = [b for b in bugs if b.resolved]
        resolved.sort(key=lambda b: b.resolved_at or "", reverse=True)
        return unresolved + resolved[:cfg.max_resolved_bugs]


# ─────────────────────────────────────────────
# Utility: relevance score for new items
# ─────────────────────────────────────────────


def score_new_item(
    content: str,
    kind: MemoryItemKind,
    priority: Priority = Priority.MEDIUM,
) -> float:
    """
    Heuristic relevance score for a freshly created memory item.
    All new items start high; decay + compression will sort them out over time.
    """
    base = 0.85
    kind_boost = {
        MemoryItemKind.BUG: 0.10,
        MemoryItemKind.CONSTRAINT: 0.08,
        MemoryItemKind.DECISION: 0.05,
        MemoryItemKind.MODULE: 0.03,
        MemoryItemKind.INSIGHT: 0.02,
        MemoryItemKind.DEPENDENCY: 0.04,
    }.get(kind, 0.0)
    priority_boost = {
        Priority.LOW: -0.10,
        Priority.MEDIUM: 0.0,
        Priority.HIGH: 0.05,
        Priority.CRITICAL: 0.12,
    }.get(priority, 0.0)
    return round(min(1.0, max(0.0, base + kind_boost + priority_boost)), 4)
