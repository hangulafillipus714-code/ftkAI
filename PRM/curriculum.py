"""
Curriculum Scheduler
====================
Implements the "Curriculum & Difficulty Scaling" idea from the design doc:

    Simple → Moderate → Hard → Abstract

Rather than random internet-mix training, this module:
  1. Assigns a difficulty score to each training example
  2. Progressively increases difficulty as training proceeds
  3. Integrates with PRM to adapt difficulty based on correction success rate
  4. Supports multi-axis difficulty (reasoning depth, context length, tool use count)

This is model-agnostic — it works as a data sampler wrapper around any
PyTorch Dataset or plain Python list.

Usage:
    scheduler = CurriculumScheduler(examples, scorer=CodeDifficultyScorer())
    for epoch in range(num_epochs):
        batch = scheduler.sample_batch(
            step=global_step,
            total_steps=total_training_steps,
            prm_correction_rate=ctrl.status()["correction_success_rate"],
        )
"""

from __future__ import annotations

import hashlib
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Optional, Sequence, Tuple, TypeVar

T = TypeVar("T")


# ─────────────────────────────────────────────
# Difficulty axes
# ─────────────────────────────────────────────


@dataclass
class DifficultyAxes:
    """
    Multi-dimensional difficulty score for one training example.
    All axes normalised to [0, 1].
    """
    reasoning_depth: float = 0.5    # how many reasoning steps required
    context_length: float = 0.5     # relative to max sequence length
    tool_use_count: float = 0.0     # number of external tool calls needed
    abstraction: float = 0.5        # concrete (0) → abstract (1)
    error_recovery: float = 0.0     # does the solution involve fixing a mistake?

    @property
    def composite(self) -> float:
        """Weighted composite score in [0, 1]."""
        return (
            0.30 * self.reasoning_depth
            + 0.20 * self.context_length
            + 0.15 * self.tool_use_count
            + 0.20 * self.abstraction
            + 0.15 * self.error_recovery
        )


@dataclass
class ScoredExample(Generic[T]):
    example: T
    difficulty: DifficultyAxes
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            # Stable hash of the repr as a default ID
            self.id = hashlib.md5(repr(self.example).encode()).hexdigest()[:8]


# ─────────────────────────────────────────────
# Difficulty Scorers
# ─────────────────────────────────────────────


class DifficultyScorer(ABC, Generic[T]):
    """Base class for difficulty scorers."""

    @abstractmethod
    def score(self, example: T) -> DifficultyAxes:
        ...

    def score_batch(self, examples: Sequence[T]) -> List[ScoredExample[T]]:
        return [ScoredExample(ex, self.score(ex)) for ex in examples]


class HeuristicCodeScorer(DifficultyScorer[str]):
    """
    Heuristic difficulty scorer for code generation examples.
    Input: a string containing the problem description + reference solution.
    """

    def score(self, example: str) -> DifficultyAxes:
        text = example.lower()
        lines = example.splitlines()

        # Reasoning depth: longer solutions → deeper reasoning
        depth = min(1.0, len(lines) / 200)

        # Context length: character count proxy
        ctx = min(1.0, len(example) / 8192)

        # Tool use: mentions of external calls
        tool_keywords = ("import", "subprocess", "requests", "open(", "os.", "sys.")
        tool_use = min(1.0, sum(text.count(k) for k in tool_keywords) / 10)

        # Abstraction: design patterns, classes, protocols
        abstract_keywords = ("class ", "abstract", "interface", "protocol", "generic", "metaclass")
        abstraction = min(1.0, sum(text.count(k) for k in abstract_keywords) / 5)

        # Error recovery: does it contain bug-fix or retry logic?
        recovery_keywords = ("fix", "retry", "fallback", "except", "error", "try:")
        error_recovery = min(1.0, sum(text.count(k) for k in recovery_keywords) / 5)

        return DifficultyAxes(
            reasoning_depth=round(depth, 3),
            context_length=round(ctx, 3),
            tool_use_count=round(tool_use, 3),
            abstraction=round(abstraction, 3),
            error_recovery=round(error_recovery, 3),
        )


class ManualScorer(DifficultyScorer[T]):
    """Use a pre-computed difficulty label (int 1–5 or float 0–1)."""

    def __init__(self, label_fn: Callable[[T], float]) -> None:
        self._label_fn = label_fn

    def score(self, example: T) -> DifficultyAxes:
        raw = max(0.0, min(1.0, float(self._label_fn(example))))
        return DifficultyAxes(
            reasoning_depth=raw,
            context_length=raw,
            abstraction=raw,
        )


# ─────────────────────────────────────────────
# Schedule shapes
# ─────────────────────────────────────────────


class ScheduleShape:
    """Controls how target difficulty grows with training progress."""

    @staticmethod
    def linear(progress: float) -> float:
        """Uniform ramp from 0 → 1."""
        return progress

    @staticmethod
    def sqrt(progress: float) -> float:
        """Slow start, fast finish."""
        return math.sqrt(progress)

    @staticmethod
    def sigmoid(progress: float, steepness: float = 8.0) -> float:
        """S-curve — fast middle, slow ends."""
        return 1.0 / (1.0 + math.exp(-steepness * (progress - 0.5)))

    @staticmethod
    def step(progress: float, n_steps: int = 4) -> float:
        """Discrete difficulty levels: Simple/Moderate/Hard/Abstract."""
        return math.floor(progress * n_steps) / n_steps

    @staticmethod
    def cosine_warmup(progress: float, warmup_frac: float = 0.05) -> float:
        """Very easy warmup, then cosine ramp to full difficulty."""
        if progress < warmup_frac:
            return progress / warmup_frac * 0.15   # stay very easy during warmup
        adjusted = (progress - warmup_frac) / (1.0 - warmup_frac)
        return 0.15 + 0.85 * (1 - math.cos(math.pi * adjusted)) / 2


# ─────────────────────────────────────────────
# Curriculum Scheduler
# ─────────────────────────────────────────────


@dataclass
class CurriculumConfig:
    # Width of the difficulty window at any point in training
    # e.g. 0.2 means only examples within ±0.1 of target are eligible
    difficulty_window: float = 0.20

    # Shape function: progress (0→1) → target_difficulty (0→1)
    schedule_shape: Callable[[float], float] = field(
        default_factory=lambda: ScheduleShape.cosine_warmup
    )

    # Minimum fraction of "easier" examples to keep in every batch (prevents forgetting)
    easy_retention_frac: float = 0.10

    # If PRM correction rate drops below this, difficulty is reduced automatically
    correction_rate_floor: float = 0.50

    # Max difficulty reduction when correction rate is poor
    correction_rate_penalty: float = 0.15

    # Reproducible sampling
    seed: Optional[int] = None


class CurriculumScheduler(Generic[T]):
    """
    Wraps a dataset of ScoredExamples and samples batches according to
    a progressive difficulty schedule.

    Optionally adapts difficulty based on PRM correction success rate:
    if the model is struggling (low correction rate), difficulty is reduced.
    """

    def __init__(
        self,
        examples: Sequence[ScoredExample[T]],
        config: Optional[CurriculumConfig] = None,
    ) -> None:
        if not examples:
            raise ValueError("examples must be non-empty")
        self._examples = list(examples)
        self._config = config or CurriculumConfig()
        self._rng = random.Random(self._config.seed)

        # Pre-sort by composite difficulty for efficient window queries
        self._sorted = sorted(self._examples, key=lambda e: e.difficulty.composite)
        self._composites = [e.difficulty.composite for e in self._sorted]

    @classmethod
    def from_raw(
        cls,
        examples: Sequence[T],
        scorer: DifficultyScorer[T],
        config: Optional[CurriculumConfig] = None,
    ) -> "CurriculumScheduler[T]":
        """Score raw examples and build a scheduler."""
        scored = scorer.score_batch(examples)
        return cls(scored, config)

    # ── Public API ───────────────────────────

    def target_difficulty(
        self,
        step: int,
        total_steps: int,
        prm_correction_rate: Optional[float] = None,
    ) -> float:
        """
        Compute the target difficulty for the current training step.

        Args:
            step:                  Current global training step.
            total_steps:           Total planned training steps.
            prm_correction_rate:   Optional: PRM correction success rate.
                                   If below floor, difficulty is reduced.
        """
        progress = min(1.0, step / max(1, total_steps))
        target = self._config.schedule_shape(progress)

        # PRM feedback: if correction rate is poor, ease off
        if prm_correction_rate is not None:
            floor = self._config.correction_rate_floor
            if prm_correction_rate < floor:
                deficit = (floor - prm_correction_rate) / floor
                penalty = deficit * self._config.correction_rate_penalty
                target = max(0.0, target - penalty)

        return round(target, 4)

    def sample_batch(
        self,
        batch_size: int,
        step: int,
        total_steps: int,
        prm_correction_rate: Optional[float] = None,
    ) -> List[ScoredExample[T]]:
        """
        Sample a batch of examples appropriate for the current difficulty target.

        A fraction of easy examples (`easy_retention_frac`) is always included
        to prevent catastrophic forgetting on simpler tasks.
        """
        cfg = self._config
        target = self.target_difficulty(step, total_steps, prm_correction_rate)

        easy_count = round(batch_size * cfg.easy_retention_frac)
        easy_count = max(0, min(easy_count, batch_size - 1))   # 0 when frac=0
        main_count = batch_size - easy_count

        # Main sample: within difficulty window around target
        main_pool = self._window(target, cfg.difficulty_window)
        if not main_pool:
            # Fallback: relax window
            main_pool = self._window(target, 0.5)
        if not main_pool:
            main_pool = self._sorted

        main_batch = self._rng.choices(main_pool, k=main_count)

        # Easy retention sample: bottom 20% difficulty
        easy_pool = self._sorted[: max(1, len(self._sorted) // 5)]
        easy_batch = self._rng.choices(easy_pool, k=easy_count)

        batch = main_batch + easy_batch
        self._rng.shuffle(batch)
        return batch

    def difficulty_distribution(self) -> Dict[str, int]:
        """
        Bucket all examples into Simple/Moderate/Hard/Abstract.
        Useful for inspecting dataset balance.
        """
        buckets: Dict[str, int] = {
            "Simple (0.00–0.25)": 0,
            "Moderate (0.25–0.50)": 0,
            "Hard (0.50–0.75)": 0,
            "Abstract (0.75–1.00)": 0,
        }
        for ex in self._examples:
            d = ex.difficulty.composite
            if d < 0.25:
                buckets["Simple (0.00–0.25)"] += 1
            elif d < 0.50:
                buckets["Moderate (0.25–0.50)"] += 1
            elif d < 0.75:
                buckets["Hard (0.50–0.75)"] += 1
            else:
                buckets["Abstract (0.75–1.00)"] += 1
        return buckets

    def progress_summary(
        self, step: int, total_steps: int, prm_correction_rate: Optional[float] = None
    ) -> str:
        target = self.target_difficulty(step, total_steps, prm_correction_rate)
        pct = step / max(1, total_steps) * 100
        pool_size = len(self._window(target, self._config.difficulty_window))
        lines = [
            f"── Curriculum Status ───────────────────",
            f"  Step        : {step:,} / {total_steps:,}  ({pct:.1f}%)",
            f"  Target diff : {target:.3f}",
            f"  Eligible    : {pool_size} examples in window",
        ]
        if prm_correction_rate is not None:
            lines.append(f"  Correction  : {prm_correction_rate:.1%}")
        lines.append("────────────────────────────────────────")
        return "\n".join(lines)

    # ── Private helpers ──────────────────────

    def _window(self, target: float, width: float) -> List[ScoredExample[T]]:
        """Return examples whose composite difficulty is within ±width/2 of target."""
        lo, hi = target - width / 2, target + width / 2
        # Binary search into sorted list
        left  = self._bisect_left(lo)
        right = self._bisect_right(hi)
        return self._sorted[left:right]

    def _bisect_left(self, val: float) -> int:
        lo, hi = 0, len(self._composites)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._composites[mid] < val:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _bisect_right(self, val: float) -> int:
        lo, hi = 0, len(self._composites)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._composites[mid] <= val:
                lo = mid + 1
            else:
                hi = mid
        return lo


# ─────────────────────────────────────────────
# PyTorch Dataset adapter (optional)
# ─────────────────────────────────────────────

try:
    from torch.utils.data import Dataset, Sampler
    import torch

    class CurriculumDataset(Dataset):
        """
        Wraps a list of scored examples as a PyTorch Dataset.
        Use with CurriculumSampler for step-aware batching.
        """

        def __init__(self, scored: Sequence[ScoredExample]) -> None:
            self._scored = list(scored)

        def __len__(self) -> int:
            return len(self._scored)

        def __getitem__(self, idx: int) -> Any:
            return self._scored[idx].example

        def difficulties(self) -> torch.Tensor:
            return torch.tensor([e.difficulty.composite for e in self._scored])

    class CurriculumSampler(Sampler):
        """
        PyTorch-compatible sampler that uses CurriculumScheduler to select
        example indices for each batch.

        Usage:
            dataset = CurriculumDataset(scored_examples)
            sampler = CurriculumSampler(
                dataset=dataset,
                scheduler=scheduler,
                batch_size=32,
                total_steps=100_000,
            )
            loader = DataLoader(dataset, batch_sampler=sampler)
        """

        def __init__(
            self,
            dataset: CurriculumDataset,
            scheduler: CurriculumScheduler,
            batch_size: int,
            total_steps: int,
            prm_correction_rate_fn: Optional[Callable[[], float]] = None,
            rank: int = 0,
            world_size: int = 1,
        ) -> None:
            self._dataset = dataset
            self._scheduler = scheduler
            # Multiply raw requested batch size by world size so _scheduler extracts enough tokens to partition properly 
            self._batch_size = batch_size * world_size
            self._total_steps = total_steps
            self._correction_rate_fn = prm_correction_rate_fn
            self._step = 0
            self._rank = rank
            self._world_size = world_size

        def __iter__(self):
            while self._step < self._total_steps:
                rate = self._correction_rate_fn() if self._correction_rate_fn else None
                batch = self._scheduler.sample_batch(
                    batch_size=self._batch_size,
                    step=self._step,
                    total_steps=self._total_steps,
                    prm_correction_rate=rate,
                )
                # Map back to dataset indices
                id_to_idx = {e.id: i for i, e in enumerate(self._scheduler._examples)}
                indices = [id_to_idx.get(e.id, 0) for e in batch]
                
                # Split indices across ranks for DDP exact matching without doubling training size
                if self._world_size > 1:
                    indices = indices[self._rank :: self._world_size]
                    
                yield indices
                self._step += 1

        def __len__(self) -> int:
            return self._total_steps

except ImportError:
    pass   # PyTorch not available — sampler classes not defined
