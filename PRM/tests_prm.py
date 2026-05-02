"""
PRM Test Suite
Validates schema, store, updater, compressor, injection, and controller.
Run: python -m pytest tests_prm.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

# ── Minimal sys.path fix so tests can import prm directly ──
import sys
sys.path.insert(0, str(Path(__file__).parent))

from prm import (
    BugReport,
    CompressionConfig,
    InjectionFormat,
    MemoryItemKind,
    PRMCompressor,
    PRMController,
    PRMState,
    PRMStore,
    PRMUpdater,
    Priority,
    ScoredItem,
    TaskNode,
    TaskStatus,
    UpdateDelta,
    build_prompt_block,
    create_project,
    score_new_item,
)


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────


@pytest.fixture
def tmp_store(tmp_path):
    store = PRMStore(tmp_path / "test.db")
    yield store
    store.close()


@pytest.fixture
def base_state():
    return PRMState(
        project_name="test_project",
        goal="Build a production LLM",
        constraints=["Must run on 2× A100"],
    )


@pytest.fixture
def rich_state(base_state):
    """State pre-populated with tasks, bugs, and memory."""
    state = base_state

    task1 = TaskNode(title="Data pipeline", priority=Priority.HIGH)
    task1.next_action = "Implement streaming loader"
    state.tasks[task1.id] = task1

    task2 = TaskNode(title="Training loop", priority=Priority.MEDIUM, depends_on=[task1.id])
    state.tasks[task2.id] = task2

    bug = BugReport(summary="OOM on batch > 4", severity=Priority.HIGH, location="train.py:42")
    state.bugs.append(bug)

    mem = ScoredItem(
        kind=MemoryItemKind.DECISION,
        content="Use streaming DataLoader to avoid OOM",
        relevance=0.9,
        confidence=0.95,
    )
    state.memory.append(mem)

    return state


# ─────────────────────────────────────────────
# Schema tests
# ─────────────────────────────────────────────


class TestSchema:
    def test_state_creation(self, base_state):
        assert base_state.project_name == "test_project"
        assert base_state.goal == "Build a production LLM"
        assert base_state.version == 1

    def test_touch_increments_version(self, base_state):
        v0 = base_state.version
        base_state.touch()
        assert base_state.version == v0 + 1
        assert base_state.total_updates == 1

    def test_add_task(self, base_state):
        t = TaskNode(title="My Task")
        base_state.add_task(t)
        assert t.id in base_state.tasks

    def test_add_bug(self, base_state):
        b = BugReport(summary="test bug")
        base_state.add_bug(b)
        assert b in base_state.bugs
        assert len(base_state.unresolved_bugs()) == 1

    def test_resolve_bug(self, base_state):
        b = BugReport(summary="OOM crash")
        base_state.add_bug(b)
        b.resolve("Added gradient checkpointing")
        assert b.resolved
        assert len(base_state.unresolved_bugs()) == 0

    def test_cycle_detection(self):
        """Task graph cycles must raise ValueError."""
        state = PRMState()
        t1 = TaskNode(title="A")
        t2 = TaskNode(title="B")
        state.tasks[t1.id] = t1
        state.tasks[t2.id] = t2
        # Create cycle: t1 → t2 → t1
        state.tasks[t1.id].depends_on = [t2.id]
        state.tasks[t2.id].depends_on = [t1.id]
        with pytest.raises(ValueError, match="Cycle detected"):
            PRMState.model_validate(state.model_dump())

    def test_active_tasks(self, rich_state):
        active = rich_state.active_tasks()
        assert all(t.status not in (TaskStatus.DONE, TaskStatus.ARCHIVED) for t in active)

    def test_summary_stats(self, rich_state):
        stats = rich_state.summary_stats()
        assert stats["total_tasks"] == 2
        assert stats["unresolved_bugs"] == 1

    def test_scored_item_decay(self):
        item = ScoredItem(kind=MemoryItemKind.INSIGHT, content="test", relevance=1.0)
        item.decay(0.5)
        assert item.relevance == 0.5

    def test_next_actions(self, rich_state):
        actions = rich_state.next_actions()
        assert any("streaming" in a for a in actions)


# ─────────────────────────────────────────────
# Store tests
# ─────────────────────────────────────────────


class TestStore:
    def test_save_and_load(self, tmp_store, base_state):
        tmp_store.save(base_state)
        loaded = tmp_store.load(base_state.project_id)
        assert loaded is not None
        assert loaded.project_name == base_state.project_name

    def test_load_missing_returns_none(self, tmp_store):
        assert tmp_store.load("nonexistent-id") is None

    def test_load_by_name(self, tmp_store, base_state):
        tmp_store.save(base_state)
        loaded = tmp_store.load_by_name("test_project")
        assert loaded is not None
        assert loaded.project_id == base_state.project_id

    def test_version_conflict_raises(self, tmp_store, base_state):
        tmp_store.save(base_state)
        # Simulate a state that has been updated externally
        newer = PRMState.model_validate_json(base_state.model_dump_json())
        newer.touch(); newer.touch(); newer.touch()   # version 4
        tmp_store.save(newer)
        # Try to save old state (version 1) — should fail
        with pytest.raises(ValueError, match="version"):
            tmp_store.save(base_state)

    def test_history_is_kept(self, tmp_store, base_state):
        tmp_store.save(base_state)
        base_state.touch()
        tmp_store.save(base_state)
        versions = tmp_store.history_versions(base_state.project_id)
        assert len(versions) >= 2

    def test_load_specific_version(self, tmp_store, base_state):
        v1 = base_state.version
        tmp_store.save(base_state)
        base_state.touch()
        tmp_store.save(base_state)
        restored = tmp_store.load_version(base_state.project_id, v1)
        assert restored is not None
        assert restored.version == v1

    def test_list_projects(self, tmp_store, base_state):
        tmp_store.save(base_state)
        projects = tmp_store.list_projects()
        assert any(p["project_id"] == base_state.project_id for p in projects)

    def test_export_import_json(self, tmp_store, base_state, tmp_path):
        tmp_store.save(base_state)
        export_path = tmp_path / "export.json"
        tmp_store.export_json(base_state.project_id, export_path)
        assert export_path.exists()
        imported = tmp_store.import_json(export_path)
        assert imported.project_id == base_state.project_id

    def test_delete(self, tmp_store, base_state):
        tmp_store.save(base_state)
        tmp_store.delete(base_state.project_id)
        assert tmp_store.load(base_state.project_id) is None


# ─────────────────────────────────────────────
# Compressor tests
# ─────────────────────────────────────────────


class TestCompressor:
    def test_decay_reduces_relevance(self, rich_state):
        original_rel = rich_state.memory[0].relevance
        cfg = CompressionConfig(decay_factor=0.5, prune_relevance_threshold=0.0)
        compressed = PRMCompressor(cfg).run(rich_state)
        assert compressed.memory[0].relevance < original_rel

    def test_low_relevance_items_pruned(self, base_state):
        item = ScoredItem(
            kind=MemoryItemKind.INSIGHT,
            content="disposable insight",
            relevance=0.01,
        )
        base_state.memory.append(item)
        cfg = CompressionConfig(prune_relevance_threshold=0.10, decay_factor=1.0)
        compressed = PRMCompressor(cfg).run(base_state)
        assert not any(i.content == "disposable insight" for i in compressed.memory)

    def test_high_relevance_items_kept(self, base_state):
        item = ScoredItem(
            kind=MemoryItemKind.CONSTRAINT,
            content="Critical: never use float16 for embedding layer",
            relevance=0.95,
        )
        base_state.memory.append(item)
        compressed = PRMCompressor().run(base_state)
        assert any("float16" in i.content for i in compressed.memory)

    def test_dedup_removes_exact_duplicates(self, base_state):
        for _ in range(5):
            base_state.memory.append(
                ScoredItem(kind=MemoryItemKind.INSIGHT, content="same content", relevance=0.8)
            )
        compressed = PRMCompressor().run(base_state)
        count = sum(1 for i in compressed.memory if i.content == "same content")
        assert count == 1

    def test_compression_counter_increments(self, base_state):
        c1 = PRMCompressor().run(base_state)
        assert c1.compression_runs == 1
        c2 = PRMCompressor().run(c1)
        assert c2.compression_runs == 2

    def test_score_new_item(self):
        bug_score = score_new_item("crash in forward pass", MemoryItemKind.BUG, Priority.CRITICAL)
        insight_score = score_new_item("minor observation", MemoryItemKind.INSIGHT, Priority.LOW)
        assert bug_score > insight_score


# ─────────────────────────────────────────────
# Injection tests
# ─────────────────────────────────────────────


class TestInjection:
    def test_text_format_contains_goal(self, rich_state):
        block = build_prompt_block(rich_state, fmt=InjectionFormat.TEXT)
        assert rich_state.goal in block

    def test_text_format_contains_bug(self, rich_state):
        block = build_prompt_block(rich_state, fmt=InjectionFormat.TEXT)
        assert "OOM" in block

    def test_json_format_is_parseable(self, rich_state):
        block = build_prompt_block(rich_state, fmt=InjectionFormat.JSON)
        data = json.loads(block)
        assert "goal" in data

    def test_xml_format_contains_tags(self, rich_state):
        block = build_prompt_block(rich_state, fmt=InjectionFormat.XML)
        assert "<project_state>" in block
        assert "</project_state>" in block

    def test_max_chars_truncation(self, rich_state):
        block = build_prompt_block(rich_state, max_chars=100)
        assert len(block) <= 150   # allow for truncation message overhead

    def test_exclude_bugs(self, rich_state):
        block = build_prompt_block(rich_state, include_bugs=False, include_memory=False)
        assert "OOM" not in block

    def test_exclude_memory(self, rich_state):
        block = build_prompt_block(rich_state, include_memory=False)
        # Memory items should not appear
        assert "streaming DataLoader" not in block


# ─────────────────────────────────────────────
# Updater tests
# ─────────────────────────────────────────────


class TestUpdater:
    def test_structured_update_stage(self, base_state):
        updater = PRMUpdater()
        output = '```json\n{"prm_update": {"stage": "training loop"}}\n```'
        new = updater.apply(base_state, output, mode="structured")
        assert new.current_stage == "training loop"

    def test_structured_update_adds_memory(self, base_state):
        updater = PRMUpdater()
        output = json.dumps({
            "prm_update": {
                "new_memory": [{"kind": "decision", "content": "Use bfloat16", "confidence": 0.9}]
            }
        })
        new = updater.apply(base_state, output, mode="structured")
        assert any("bfloat16" in m.content for m in new.memory)

    def test_structured_update_adds_bug(self, base_state):
        updater = PRMUpdater()
        output = json.dumps({
            "prm_update": {
                "new_bugs": [{"summary": "NaN loss after 1000 steps", "severity": "high"}]
            }
        })
        new = updater.apply(base_state, output, mode="structured")
        assert any("NaN" in b.summary for b in new.bugs)

    def test_structured_update_resolves_bug(self, base_state):
        bug = BugReport(summary="NaN loss")
        base_state.bugs.append(bug)
        updater = PRMUpdater()
        output = json.dumps({
            "prm_update": {
                "resolved_bug_ids": [bug.id],
                "resolved_bug_fixes": {bug.id: "Added loss scaling"}
            }
        })
        new = updater.apply(base_state, output, mode="structured")
        resolved = next(b for b in new.bugs if b.id == bug.id)
        assert resolved.resolved
        assert "loss scaling" in resolved.fix_description

    def test_heuristic_extracts_bug(self, base_state):
        updater = PRMUpdater()
        output = "I found a bug: memory leak in the attention mask computation"
        new = updater.apply(base_state, output, mode="heuristic")
        assert len(new.bugs) >= 1

    def test_heuristic_extracts_decision(self, base_state):
        updater = PRMUpdater()
        output = "I decided to use flash attention for memory efficiency"
        new = updater.apply(base_state, output, mode="heuristic")
        assert any("flash attention" in m.content for m in new.memory)

    def test_auto_mode_structured_first(self, base_state):
        updater = PRMUpdater()
        output = '{"prm_update": {"stage": "eval"}}'
        new = updater.apply(base_state, output, mode="auto")
        assert new.current_stage == "eval"

    def test_auto_mode_falls_back_to_heuristic(self, base_state):
        updater = PRMUpdater()
        output = "The bug is in the tokenizer padding"
        new = updater.apply(base_state, output, mode="auto")
        # Heuristic should pick up the bug mention
        assert new.total_updates > base_state.total_updates

    def test_passthrough_delta(self, base_state):
        updater = PRMUpdater()
        delta = UpdateDelta(stage="data loading")
        new = updater.apply_delta(base_state, delta)
        assert new.current_stage == "data loading"

    def test_task_status_update(self, base_state):
        t = TaskNode(title="Preprocess data")
        base_state.tasks[t.id] = t
        updater = PRMUpdater()
        delta = UpdateDelta(task_updates=[
            __import__("prm.updater", fromlist=["TaskUpdate"]).TaskUpdate(
                task_id=t.id, status="done", next_action="Proceed to training"
            )
        ])
        new = updater.apply_delta(base_state, delta)
        assert new.tasks[t.id].status == TaskStatus.DONE


# ─────────────────────────────────────────────
# Controller tests
# ─────────────────────────────────────────────


class TestController:
    def _echo_model(self, prompt: str) -> str:
        return '{"prm_update": {"stage": "step_1"}}'

    def test_full_cycle(self, tmp_store, base_state):
        tmp_store.save(base_state)
        ctrl = PRMController(
            model_fn=self._echo_model,
            store=tmp_store,
            project_id=base_state.project_id,
        )
        result = ctrl.step("Start training")
        assert result.model_output
        assert result.updated_state.current_stage == "step_1"
        assert result.latency_ms > 0

    def test_state_is_persisted(self, tmp_store, base_state):
        tmp_store.save(base_state)
        ctrl = PRMController(
            model_fn=self._echo_model,
            store=tmp_store,
            project_id=base_state.project_id,
        )
        ctrl.step("Do something")
        reloaded = tmp_store.load(base_state.project_id)
        assert reloaded.current_stage == "step_1"

    def test_status(self, tmp_store, base_state):
        tmp_store.save(base_state)
        ctrl = PRMController(
            model_fn=self._echo_model,
            store=tmp_store,
            project_id=base_state.project_id,
        )
        status = ctrl.status()
        assert "project_name" in status
        assert status["project_name"] == "test_project"

    def test_model_error_returns_error_field(self, tmp_store, base_state):
        tmp_store.save(base_state)

        def bad_model(prompt: str) -> str:
            raise RuntimeError("GPU OOM")

        from prm.controller import ControllerConfig
        cfg = ControllerConfig(max_retries=0, auto_save=False)
        ctrl = PRMController(
            model_fn=bad_model,
            store=tmp_store,
            project_id=base_state.project_id,
            config=cfg,
        )
        result = ctrl.step("Train")
        assert result.error is not None
        assert "OOM" in result.error

    def test_create_project_helper(self, tmp_store):
        state = create_project(tmp_store, "new_proj", "Train a model")
        loaded = tmp_store.load(state.project_id)
        assert loaded is not None
        assert loaded.goal == "Train a model"

    def test_force_compress(self, tmp_store, base_state):
        # Populate with many items
        for i in range(30):
            base_state.memory.append(
                ScoredItem(
                    kind=MemoryItemKind.INSIGHT,
                    content=f"insight {i}",
                    relevance=0.05,  # very low — should be pruned
                )
            )
        tmp_store.save(base_state)
        ctrl = PRMController(
            model_fn=self._echo_model,
            store=tmp_store,
            project_id=base_state.project_id,
        )
        compressed = ctrl.force_compress()
        # Very low relevance items should have been pruned
        assert len(compressed.memory) < 30


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
