"""
Tests for metrics, curriculum scheduler, and CLI.
Run: python -m pytest tests_prm_ext.py -v
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from prm import (
    BugReport, PRMState, PRMStore, ScoredItem, MemoryItemKind,
    TaskNode, TaskStatus, Priority,
    compute_snapshot_metrics, compute_session_metrics, diff_states,
    MetricsLogger,
    CurriculumScheduler, CurriculumConfig, DifficultyAxes,
    ScoredExample, HeuristicCodeScorer, ManualScorer, ScheduleShape,
    create_project,
)


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────


@pytest.fixture
def tmp_store(tmp_path):
    store = PRMStore(tmp_path / "test.db")
    yield store
    store.close()


def _make_state(n_tasks=3, n_done=1, n_bugs=2, n_resolved=1) -> PRMState:
    state = PRMState(project_name="test", goal="Build LLM")
    for i in range(n_tasks):
        t = TaskNode(title=f"Task {i}", priority=Priority.MEDIUM)
        if i < n_done:
            t.status = TaskStatus.DONE
        state.tasks[t.id] = t
    for i in range(n_bugs):
        b = BugReport(summary=f"Bug {i}", severity=Priority.HIGH)
        if i < n_resolved:
            b.resolve(f"Fix {i}")
        state.bugs.append(b)
    for i in range(5):
        state.memory.append(ScoredItem(
            kind=MemoryItemKind.INSIGHT,
            content=f"Insight {i}",
            relevance=0.5 + i * 0.1,
            confidence=0.8,
        ))
    return state


# ─────────────────────────────────────────────
# Metrics tests
# ─────────────────────────────────────────────


class TestSnapshotMetrics:
    def test_task_completion_rate(self):
        state = _make_state(n_tasks=4, n_done=2)
        m = compute_snapshot_metrics(state)
        assert m.task_completion_rate == pytest.approx(0.5, abs=0.01)

    def test_bug_resolution_rate(self):
        state = _make_state(n_bugs=4, n_resolved=2)
        m = compute_snapshot_metrics(state)
        assert m.bug_resolution_rate == pytest.approx(0.5, abs=0.01)

    def test_open_bug_count(self):
        state = _make_state(n_bugs=3, n_resolved=1)
        m = compute_snapshot_metrics(state)
        assert m.open_bug_count == 2

    def test_memory_stats(self):
        state = _make_state()
        m = compute_snapshot_metrics(state)
        assert m.memory_item_count == 5
        assert 0.0 <= m.avg_memory_relevance <= 1.0

    def test_high_relevance_ratio(self):
        state = PRMState(project_name="x", goal="y")
        for rel in [0.9, 0.8, 0.2, 0.1]:
            state.memory.append(ScoredItem(kind=MemoryItemKind.INSIGHT, content="x", relevance=rel))
        m = compute_snapshot_metrics(state)
        assert m.high_relevance_ratio == pytest.approx(0.5, abs=0.01)

    def test_compression_efficiency(self):
        state = _make_state()
        state.archived_item_count = 10
        m = compute_snapshot_metrics(state)
        assert m.compression_efficiency > 0


class TestSessionMetrics:
    def test_completion_trajectory(self):
        states = []
        for done in [0, 1, 2, 3]:
            s = _make_state(n_tasks=3, n_done=done)
            states.append(s)
        sm = compute_session_metrics(states)
        assert sm.task_completion_rate_final > sm.task_completion_rate_peak * 0.5

    def test_error_propagation_depth(self):
        # Bug added at v1, resolved at v3 → depth 2
        s1 = PRMState(project_name="p", goal="g")
        s1.version = 1
        b = BugReport(summary="NaN loss")
        s1.bugs.append(b)

        s2 = PRMState.model_validate_json(s1.model_dump_json())
        s2.version = 2

        s3 = PRMState.model_validate_json(s2.model_dump_json())
        s3.version = 3
        for bug in s3.bugs:
            bug.resolve("Fixed with loss scaling")

        sm = compute_session_metrics([s1, s2, s3])
        assert sm.avg_error_propagation_depth >= 0

    def test_correction_success_rate(self):
        s1 = _make_state(n_bugs=2, n_resolved=0)
        s2 = PRMState.model_validate_json(s1.model_dump_json())
        for b in s2.bugs:
            b.resolve("fixed")
        sm = compute_session_metrics([s1, s2])
        assert sm.correction_success_rate == pytest.approx(1.0)

    def test_memory_trend_positive(self):
        states = []
        for i in range(5):
            s = PRMState(project_name="p", goal="g")
            s.memory.append(ScoredItem(
                kind=MemoryItemKind.INSIGHT, content="x",
                relevance=0.5 + i * 0.1
            ))
            states.append(s)
        sm = compute_session_metrics(states)
        assert sm.avg_memory_relevance_trend > 0


class TestStateDiff:
    def test_diff_completion(self):
        before = _make_state(n_tasks=3, n_done=0)
        after = PRMState.model_validate_json(before.model_dump_json())
        for t in after.tasks.values():
            t.status = TaskStatus.DONE
        d = diff_states(before, after)
        assert d.tasks_completed == 3

    def test_diff_bugs(self):
        before = _make_state(n_bugs=0)
        after = PRMState.model_validate_json(before.model_dump_json())
        after.bugs.append(BugReport(summary="New bug"))
        d = diff_states(before, after)
        assert d.bugs_added == 1

    def test_diff_memory_pruned(self):
        before = _make_state()
        after = PRMState.model_validate_json(before.model_dump_json())
        after.memory.clear()
        d = diff_states(before, after)
        assert d.memory_pruned == 5


class TestMetricsLogger:
    def test_log_and_load(self, tmp_path):
        logger = MetricsLogger(tmp_path / "m.db")
        state = _make_state()
        m = compute_snapshot_metrics(state)
        logger.log(m)
        history = logger.load_history(state.project_id)
        assert len(history) == 1
        assert history[0].version == m.version
        logger.close()

    def test_export_csv(self, tmp_path):
        logger = MetricsLogger(tmp_path / "m.db")
        for i in range(3):
            state = _make_state()
            state.version = i + 1
            logger.log(compute_snapshot_metrics(state))
        csv_path = tmp_path / "metrics.csv"
        logger.export_csv(state.project_id, csv_path)
        assert csv_path.exists()
        content = csv_path.read_text()
        assert "task_completion_rate" in content
        logger.close()

    def test_summary_report(self, tmp_path):
        logger = MetricsLogger(tmp_path / "m.db")
        state = _make_state()
        logger.log(compute_snapshot_metrics(state))
        report = logger.summary_report(state.project_id)
        assert "Completion" in report
        logger.close()

    def test_empty_report(self, tmp_path):
        logger = MetricsLogger(tmp_path / "m.db")
        report = logger.summary_report("nonexistent")
        assert "No metrics" in report
        logger.close()


# ─────────────────────────────────────────────
# Curriculum tests
# ─────────────────────────────────────────────


def _make_scored_examples(n=100):
    examples = []
    for i in range(n):
        difficulty = i / (n - 1)
        axes = DifficultyAxes(
            reasoning_depth=difficulty,
            context_length=difficulty,
            abstraction=difficulty,
        )
        examples.append(ScoredExample(example=f"example_{i}", difficulty=axes))
    return examples


class TestCurriculumScheduler:
    def test_difficulty_window_respected(self):
        examples = _make_scored_examples(100)
        # easy_retention_frac=0 so ALL items come from the main window
        cfg = CurriculumConfig(difficulty_window=0.2, easy_retention_frac=0.0, seed=42)
        sched = CurriculumScheduler(examples, cfg)
        batch = sched.sample_batch(batch_size=20, step=500, total_steps=1000)
        target = sched.target_difficulty(500, 1000)
        # Window = ±0.1; allow a tiny float tolerance
        for ex in batch:
            assert abs(ex.difficulty.composite - target) <= 0.10 + 1e-4

    def test_easy_retention(self):
        examples = _make_scored_examples(100)
        cfg = CurriculumConfig(easy_retention_frac=0.25, seed=42)
        sched = CurriculumScheduler(examples, cfg)
        # At end of training, easy examples should still appear
        batch = sched.sample_batch(batch_size=40, step=999, total_steps=1000)
        easy = [e for e in batch if e.difficulty.composite < 0.25]
        assert len(easy) >= 1

    def test_target_increases_with_steps(self):
        examples = _make_scored_examples(50)
        sched = CurriculumScheduler(examples)
        targets = [sched.target_difficulty(s, 1000) for s in [0, 250, 500, 750, 1000]]
        # Should be non-decreasing
        for i in range(len(targets) - 1):
            assert targets[i] <= targets[i + 1] + 1e-6

    def test_prm_correction_rate_reduces_difficulty(self):
        examples = _make_scored_examples(50)
        sched = CurriculumScheduler(examples)
        target_normal = sched.target_difficulty(500, 1000, prm_correction_rate=0.9)
        target_poor   = sched.target_difficulty(500, 1000, prm_correction_rate=0.1)
        assert target_poor < target_normal

    def test_difficulty_distribution(self):
        examples = _make_scored_examples(100)
        sched = CurriculumScheduler(examples)
        dist = sched.difficulty_distribution()
        total = sum(dist.values())
        assert total == 100
        assert all(v >= 0 for v in dist.values())

    def test_from_raw_with_scorer(self):
        raw = ["short code", "x" * 500, "import torch\nclass Model(nn.Module): pass"]
        sched = CurriculumScheduler.from_raw(raw, HeuristicCodeScorer())
        assert len(sched._examples) == 3

    def test_manual_scorer(self):
        raw = [1, 2, 3, 4, 5]
        scorer = ManualScorer(lambda x: x / 5.0)
        sched = CurriculumScheduler.from_raw(raw, scorer)
        composites = sorted(e.difficulty.composite for e in sched._examples)
        assert composites[0] < composites[-1]

    def test_schedule_shapes(self):
        for shape in [
            ScheduleShape.linear,
            ScheduleShape.sqrt,
            ScheduleShape.sigmoid,
            ScheduleShape.step,
            ScheduleShape.cosine_warmup,
        ]:
            assert shape(0.0) < shape(1.0) + 1e-6
            assert 0.0 <= shape(0.5) <= 1.0

    def test_progress_summary(self):
        examples = _make_scored_examples(50)
        sched = CurriculumScheduler(examples)
        summary = sched.progress_summary(500, 1000)
        assert "Step" in summary
        assert "Target diff" in summary

    def test_sample_large_batch(self):
        examples = _make_scored_examples(200)
        sched = CurriculumScheduler(examples, CurriculumConfig(seed=0))
        batch = sched.sample_batch(batch_size=64, step=100, total_steps=10000)
        assert len(batch) == 64

    def test_empty_window_fallback(self):
        # All examples clustered at 0.0 difficulty; target is 1.0
        examples = [
            ScoredExample(example="e", difficulty=DifficultyAxes(reasoning_depth=0.0))
        ] * 10
        cfg = CurriculumConfig(difficulty_window=0.1, easy_retention_frac=0.0, seed=1)
        sched = CurriculumScheduler(examples, cfg)
        # Should not raise even when window is empty
        batch = sched.sample_batch(batch_size=5, step=999, total_steps=1000)
        assert len(batch) == 5


class TestHeuristicCodeScorer:
    def test_longer_code_is_harder(self):
        scorer = HeuristicCodeScorer()
        short = "x = 1"
        long  = "\n".join(["import torch"] * 100 + ["class Model(nn.Module): pass"] * 50)
        assert scorer.score(long).composite > scorer.score(short).composite

    def test_abstract_code_scores_higher(self):
        scorer = HeuristicCodeScorer()
        concrete = "result = a + b"
        abstract = "class AbstractFactory(Protocol, metaclass=ABCMeta): pass"
        assert scorer.score(abstract).abstraction > scorer.score(concrete).abstraction


# ─────────────────────────────────────────────
# CLI tests (via main())
# ─────────────────────────────────────────────


class TestCLI:
    def _run(self, argv, tmp_db):
        from prm.cli import main
        main(["--db", str(tmp_db)] + argv)

    def test_init_and_list(self, tmp_path, capsys):
        db = tmp_path / "cli.db"
        self._run(["init", "--name", "cli_project", "--goal", "Test goal"], db)
        self._run(["list"], db)
        out = capsys.readouterr().out
        assert "cli_project" in out

    def test_status(self, tmp_path, capsys):
        db = tmp_path / "cli.db"
        self._run(["init", "--name", "proj", "--goal", "G"], db)
        self._run(["status", "--project", "proj"], db)
        out = capsys.readouterr().out
        assert "Goal" in out

    def test_show_json(self, tmp_path, capsys):
        db = tmp_path / "cli.db"
        self._run(["init", "--name", "proj", "--goal", "G"], db)
        capsys.readouterr()   # clear the [ok] init message
        self._run(["show", "--project", "proj", "--format", "json"], db)
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "goal" in data

    def test_show_xml(self, tmp_path, capsys):
        db = tmp_path / "cli.db"
        self._run(["init", "--name", "proj", "--goal", "G"], db)
        self._run(["show", "--project", "proj", "--format", "xml"], db)
        out = capsys.readouterr().out
        assert "<project_state>" in out

    def test_task_add_and_done(self, tmp_path, capsys):
        db = tmp_path / "cli.db"
        self._run(["init", "--name", "proj", "--goal", "G"], db)
        self._run(["task", "add", "--project", "proj", "--title", "My Task", "--priority", "high"], db)
        out = capsys.readouterr().out
        task_id = out.split("id=")[-1].strip()
        self._run(["task", "done", "--project", "proj", "--task-id", task_id], db)
        capsys.readouterr()
        self._run(["status", "--project", "proj"], db)
        out = capsys.readouterr().out
        assert "1 done" in out.replace(" ", " ")

    def test_bug_add_and_resolve(self, tmp_path, capsys):
        db = tmp_path / "cli.db"
        self._run(["init", "--name", "proj", "--goal", "G"], db)
        self._run(["bug", "add", "--project", "proj",
                   "--summary", "Test crash", "--severity", "high"], db)
        out = capsys.readouterr().out
        bug_id = out.split("id=")[-1].strip()
        self._run(["bug", "resolve", "--project", "proj", "--bug-id", bug_id, "--fix", "Patched"], db)
        capsys.readouterr()
        self._run(["status", "--project", "proj"], db)
        out = capsys.readouterr().out
        assert "0 open" in out

    def test_memory_add(self, tmp_path, capsys):
        db = tmp_path / "cli.db"
        self._run(["init", "--name", "proj", "--goal", "G"], db)
        self._run(["memory", "add", "--project", "proj",
                   "--kind", "decision", "--content", "Use bfloat16"], db)
        capsys.readouterr()
        self._run(["show", "--project", "proj", "--format", "json"], db)
        out = capsys.readouterr().out
        data = json.loads(out)
        assert any("bfloat16" in m["content"] for m in data.get("memory", []))

    def test_compress(self, tmp_path, capsys):
        db = tmp_path / "cli.db"
        self._run(["init", "--name", "proj", "--goal", "G"], db)
        self._run(["compress", "--project", "proj"], db)
        out = capsys.readouterr().out
        assert "Compression run" in out

    def test_export_and_import(self, tmp_path, capsys):
        db = tmp_path / "cli.db"
        out_json = tmp_path / "export.json"
        self._run(["init", "--name", "proj", "--goal", "G"], db)
        self._run(["export", "--project", "proj", "--output", str(out_json)], db)
        assert out_json.exists()
        db2 = tmp_path / "cli2.db"
        self._run(["import", "--file", str(out_json)], db2)
        capsys.readouterr()
        self._run(["list"], db2)
        out = capsys.readouterr().out
        assert "proj" in out

    def test_history(self, tmp_path, capsys):
        db = tmp_path / "cli.db"
        self._run(["init", "--name", "proj", "--goal", "G"], db)
        self._run(["history", "--project", "proj"], db)
        out = capsys.readouterr().out
        assert "History" in out

    def test_rollback(self, tmp_path, capsys):
        db = tmp_path / "cli.db"
        self._run(["init", "--name", "proj", "--goal", "G"], db)
        # Get initial version
        from prm import PRMStore
        store = PRMStore(db)
        state = store.load_by_name("proj")
        v1 = state.version
        state.touch(); store.save(state)
        store.close()
        self._run(["rollback", "--project", "proj", "--version", str(v1)], db)
        out = capsys.readouterr().out
        assert "Rolled back" in out

    def test_metrics_command(self, tmp_path, capsys):
        db = tmp_path / "cli.db"
        self._run(["init", "--name", "proj", "--goal", "G"], db)
        self._run(["metrics", "--project", "proj"], db)
        out = capsys.readouterr().out
        assert "Task completion" in out

    def test_init_duplicate_fails(self, tmp_path):
        db = tmp_path / "cli.db"
        self._run(["init", "--name", "dup", "--goal", "G"], db)
        with pytest.raises(SystemExit):
            self._run(["init", "--name", "dup", "--goal", "G"], db)

    def test_init_force_overwrites(self, tmp_path, capsys):
        db = tmp_path / "cli.db"
        self._run(["init", "--name", "dup", "--goal", "G"], db)
        self._run(["init", "--name", "dup", "--goal", "G2", "--force"], db)
        capsys.readouterr()
        self._run(["list"], db)
        out = capsys.readouterr().out
        assert "dup" in out


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
