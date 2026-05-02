"""
PRM CLI — Command-Line Interface
=================================
Manage PRM projects, inspect state, run compression, roll back versions,
and view metrics — all from the terminal.

Usage:
    python -m prm.cli --help
    python -m prm.cli init   --name my_project --goal "Train a 1B LLM"
    python -m prm.cli status --project my_project
    python -m prm.cli show   --project my_project --format text
    python -m prm.cli compress --project my_project
    python -m prm.cli rollback --project my_project --version 5
    python -m prm.cli list
    python -m prm.cli export --project my_project --output ./backup.json
    python -m prm.cli import --file ./backup.json
    python -m prm.cli task add   --project my_project --title "Data pipeline" --priority high
    python -m prm.cli task done  --project my_project --task-id abc12345
    python -m prm.cli bug add    --project my_project --summary "OOM at batch 4"
    python -m prm.cli bug resolve --project my_project --bug-id xyz98765 --fix "Added grad checkpoint"
    python -m prm.cli memory add --project my_project --kind decision --content "Use bfloat16"
    python -m prm.cli metrics    --project my_project
    python -m prm.cli history    --project my_project
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

# ── Resolve DB path from env or default ─────
_DEFAULT_DB = os.environ.get("PRM_DB", "./prm.db")


def _get_store(db: str):
    from prm import PRMStore
    return PRMStore(db)


def _get_state(store, project_name_or_id: str):
    from prm import PRMStore
    # Try as project_id first
    state = store.load(project_name_or_id)
    if state is None:
        # Try as name
        state = store.load_by_name(project_name_or_id)
    if state is None:
        print(f"[error] Project '{project_name_or_id}' not found in {store.db_path}", file=sys.stderr)
        sys.exit(1)
    return state


# ─────────────────────────────────────────────
# Command handlers
# ─────────────────────────────────────────────


def cmd_list(args):
    store = _get_store(args.db)
    projects = store.list_projects()
    if not projects:
        print("No projects found.")
        return
    print(f"\n{'NAME':<30} {'VERSION':>7}  {'UPDATED':<26}  ID")
    print("─" * 85)
    for p in projects:
        print(f"{p['name']:<30} {p['version']:>7}  {p['updated_at']:<26}  {p['project_id']}")
    print()


def cmd_init(args):
    from prm import PRMStore, PRMState
    store = _get_store(args.db)
    existing = store.load_by_name(args.name)
    if existing and not args.force:
        print(f"[error] Project '{args.name}' already exists (id={existing.project_id}). Use --force to overwrite.")
        sys.exit(1)
    constraints = args.constraints or []
    state = PRMState(
        project_name=args.name,
        goal=args.goal or "",
        constraints=constraints,
    )
    store.save(state)
    print(f"[ok] Created project '{args.name}'  id={state.project_id}")
    store.close()


def cmd_status(args):
    store = _get_store(args.db)
    state = _get_state(store, args.project)
    stats = state.summary_stats()
    active = state.active_tasks()
    bugs = state.unresolved_bugs()
    next_a = state.next_actions()
    print(f"\n── Project: {state.project_name} ({'id=' + state.project_id})")
    print(f"  Goal     : {state.goal or '(not set)'}")
    print(f"  Stage    : {state.current_stage or '(not set)'}")
    print(f"  Version  : {stats['version']}  |  Updates: {state.total_updates}  |  Compressions: {stats['compression_runs']}")
    print(f"  Tasks    : {stats['active_tasks']} active / {stats['done_tasks']} done / {stats['total_tasks']} total")
    print(f"  Bugs     : {stats['unresolved_bugs']} open")
    print(f"  Memory   : {stats['total_memory_items']} items  (archived: {stats['archived_items']})")
    if active:
        print("\n  Active tasks:")
        for t in sorted(active, key=lambda x: x.priority, reverse=True)[:5]:
            print(f"    [{t.priority.upper():8}] {t.id}  {t.title}")
            if t.next_action:
                print(f"              → {t.next_action}")
    if bugs:
        print("\n  Open bugs:")
        for b in bugs[:5]:
            print(f"    [{b.severity.upper():8}] {b.id}  {b.summary}")
    if next_a:
        print("\n  Next actions:")
        for a in next_a[:3]:
            print(f"    {a}")
    print()
    store.close()


def cmd_show(args):
    from prm import build_prompt_block, InjectionFormat
    store = _get_store(args.db)
    state = _get_state(store, args.project)
    fmt_map = {"text": InjectionFormat.TEXT, "json": InjectionFormat.JSON, "xml": InjectionFormat.XML}
    fmt = fmt_map.get(args.format, InjectionFormat.TEXT)
    print(build_prompt_block(state, fmt=fmt))
    store.close()


def cmd_compress(args):
    from prm import PRMCompressor
    store = _get_store(args.db)
    state = _get_state(store, args.project)
    before_mem = len(state.memory)
    compressed = PRMCompressor().run(state)
    store.save(compressed)
    after_mem = len(compressed.memory)
    print(f"[ok] Compression run #{compressed.compression_runs} complete.")
    print(f"     Memory: {before_mem} → {after_mem}  (archived {state.archived_item_count} → {compressed.archived_item_count})")
    store.close()


def cmd_rollback(args):
    store = _get_store(args.db)
    state = _get_state(store, args.project)
    restored = store.load_version(state.project_id, args.version)
    if restored is None:
        print(f"[error] Version {args.version} not found for project '{args.project}'", file=sys.stderr)
        sys.exit(1)
    # Bump version above current so the save is accepted
    restored.version = state.version + 1
    store.save(restored)
    print(f"[ok] Rolled back to version {args.version} (now saved as version {restored.version})")
    store.close()


def cmd_export(args):
    store = _get_store(args.db)
    state = _get_state(store, args.project)
    out = Path(args.output)
    store.export_json(state.project_id, out)
    print(f"[ok] Exported to {out}")
    store.close()


def cmd_import(args):
    store = _get_store(args.db)
    state = store.import_json(args.file)
    print(f"[ok] Imported '{state.project_name}'  id={state.project_id}  version={state.version}")
    store.close()


def cmd_history(args):
    store = _get_store(args.db)
    state = _get_state(store, args.project)
    versions = store.history_versions(state.project_id)
    if not versions:
        print("No history found.")
        return
    print(f"\nHistory for '{state.project_name}' ({len(versions)} snapshots):")
    print(f"  {'VERSION':>8}   (rollback with: prm rollback --project {args.project} --version N)")
    for v in versions:
        marker = " ← current" if v == state.version else ""
        print(f"  {v:>8}{marker}")
    print()
    store.close()


# ── Task sub-commands ─────────────────────────


def cmd_task_add(args):
    from prm.schema import TaskNode, Priority
    store = _get_store(args.db)
    state = _get_state(store, args.project)
    pri = Priority(args.priority) if args.priority else Priority.MEDIUM
    task = TaskNode(
        title=args.title,
        description=args.description or "",
        priority=pri,
        next_action=args.next_action,
        depends_on=args.depends_on or [],
    )
    state.add_task(task)
    store.save(state)
    print(f"[ok] Added task '{args.title}'  id={task.id}")
    store.close()


def cmd_task_done(args):
    from prm.schema import TaskStatus
    store = _get_store(args.db)
    state = _get_state(store, args.project)
    if args.task_id not in state.tasks:
        print(f"[error] Task '{args.task_id}' not found", file=sys.stderr)
        sys.exit(1)
    state.tasks[args.task_id].status = TaskStatus.DONE
    state.tasks[args.task_id].touch()
    state.touch()
    store.save(state)
    print(f"[ok] Task '{args.task_id}' marked as DONE")
    store.close()


def cmd_task_update(args):
    from prm.schema import TaskStatus, Priority
    store = _get_store(args.db)
    state = _get_state(store, args.project)
    if args.task_id not in state.tasks:
        print(f"[error] Task '{args.task_id}' not found", file=sys.stderr)
        sys.exit(1)
    task = state.tasks[args.task_id]
    if args.status:
        task.status = TaskStatus(args.status)
    if args.next_action:
        task.next_action = args.next_action
    if args.priority:
        task.priority = Priority(args.priority)
    task.touch()
    state.touch()
    store.save(state)
    print(f"[ok] Task '{args.task_id}' updated")
    store.close()


# ── Bug sub-commands ──────────────────────────


def cmd_bug_add(args):
    from prm.schema import BugReport, Priority
    store = _get_store(args.db)
    state = _get_state(store, args.project)
    pri = Priority(args.severity) if args.severity else Priority.MEDIUM
    bug = BugReport(
        summary=args.summary,
        location=args.location,
        severity=pri,
    )
    state.add_bug(bug)
    store.save(state)
    print(f"[ok] Added bug '{args.summary}'  id={bug.id}")
    store.close()


def cmd_bug_resolve(args):
    store = _get_store(args.db)
    state = _get_state(store, args.project)
    found = False
    for bug in state.bugs:
        if bug.id == args.bug_id:
            bug.resolve(args.fix or "resolved")
            found = True
            break
    if not found:
        print(f"[error] Bug '{args.bug_id}' not found", file=sys.stderr)
        sys.exit(1)
    state.touch()
    store.save(state)
    print(f"[ok] Bug '{args.bug_id}' resolved")
    store.close()


# ── Memory sub-commands ───────────────────────


def cmd_memory_add(args):
    from prm.schema import ScoredItem, MemoryItemKind
    from prm.compressor import score_new_item
    store = _get_store(args.db)
    state = _get_state(store, args.project)
    kind = MemoryItemKind(args.kind) if args.kind else MemoryItemKind.INSIGHT
    rel = float(args.relevance) if args.relevance else score_new_item(args.content, kind)
    item = ScoredItem(
        kind=kind,
        content=args.content,
        relevance=rel,
        confidence=float(args.confidence) if args.confidence else 0.85,
        tags=args.tags or [],
    )
    state.add_memory_item(item)
    store.save(state)
    print(f"[ok] Added memory item  id={item.id}  rel={item.relevance:.2f}  kind={item.kind}")
    store.close()


# ── Metrics sub-command ───────────────────────


def cmd_metrics(args):
    from prm.metrics import compute_snapshot_metrics, MetricsLogger
    store = _get_store(args.db)
    state = _get_state(store, args.project)
    m = compute_snapshot_metrics(state)
    print(f"\n── Snapshot Metrics (v{m.version}) ─────────────────────")
    print(f"  Task completion      : {m.task_completion_rate:.1%}")
    print(f"  Task failure rate    : {m.task_failure_rate:.1%}")
    print(f"  Task blocked rate    : {m.task_blocked_rate:.1%}")
    print(f"  Open bugs            : {m.open_bug_count}  (critical: {m.critical_bug_count})")
    print(f"  Bug resolution rate  : {m.bug_resolution_rate:.1%}")
    print(f"  Avg memory relevance : {m.avg_memory_relevance:.3f}")
    print(f"  Avg memory confidence: {m.avg_memory_confidence:.3f}")
    print(f"  High-relevance ratio : {m.high_relevance_ratio:.1%}")
    print(f"  Memory items         : {m.memory_item_count}")
    print(f"  Goal confidence      : {m.goal_confidence:.2f}")
    print(f"  Compression runs     : {m.compression_runs}")
    print(f"  Items archived       : {m.archived_item_count}")
    print(f"  Compression eff.     : {m.compression_efficiency:.1%}")
    print()
    store.close()


# ─────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prm",
        description="Persistent Reasoning Memory — CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--db", default=_DEFAULT_DB, help="Path to SQLite DB (or set PRM_DB env var)")

    sub = parser.add_subparsers(dest="command")

    # list
    sub.add_parser("list", help="List all projects")

    # init
    p_init = sub.add_parser("init", help="Create a new project")
    p_init.add_argument("--name", required=True)
    p_init.add_argument("--goal", default="")
    p_init.add_argument("--constraints", nargs="*")
    p_init.add_argument("--force", action="store_true")

    # status
    p_status = sub.add_parser("status", help="Show project status")
    p_status.add_argument("--project", required=True, metavar="NAME_OR_ID")

    # show
    p_show = sub.add_parser("show", help="Show full PRM state as prompt block")
    p_show.add_argument("--project", required=True, metavar="NAME_OR_ID")
    p_show.add_argument("--format", choices=["text", "json", "xml"], default="text")

    # compress
    p_comp = sub.add_parser("compress", help="Run compression on a project")
    p_comp.add_argument("--project", required=True, metavar="NAME_OR_ID")

    # rollback
    p_rb = sub.add_parser("rollback", help="Roll back to a previous version")
    p_rb.add_argument("--project", required=True, metavar="NAME_OR_ID")
    p_rb.add_argument("--version", required=True, type=int)

    # export
    p_ex = sub.add_parser("export", help="Export project state to JSON")
    p_ex.add_argument("--project", required=True, metavar="NAME_OR_ID")
    p_ex.add_argument("--output", required=True)

    # import
    p_im = sub.add_parser("import", help="Import a JSON state file")
    p_im.add_argument("--file", required=True)

    # history
    p_hist = sub.add_parser("history", help="List stored versions for a project")
    p_hist.add_argument("--project", required=True, metavar="NAME_OR_ID")

    # metrics
    p_met = sub.add_parser("metrics", help="Show metrics for a project")
    p_met.add_argument("--project", required=True, metavar="NAME_OR_ID")

    # task
    p_task = sub.add_parser("task", help="Manage tasks")
    task_sub = p_task.add_subparsers(dest="task_command")

    p_task_add = task_sub.add_parser("add")
    p_task_add.add_argument("--project", required=True, metavar="NAME_OR_ID")
    p_task_add.add_argument("--title", required=True)
    p_task_add.add_argument("--description", default="")
    p_task_add.add_argument("--priority", choices=["low", "medium", "high", "critical"], default="medium")
    p_task_add.add_argument("--next-action")
    p_task_add.add_argument("--depends-on", nargs="*")

    p_task_done = task_sub.add_parser("done")
    p_task_done.add_argument("--project", required=True, metavar="NAME_OR_ID")
    p_task_done.add_argument("--task-id", required=True)

    p_task_upd = task_sub.add_parser("update")
    p_task_upd.add_argument("--project", required=True, metavar="NAME_OR_ID")
    p_task_upd.add_argument("--task-id", required=True)
    p_task_upd.add_argument("--status", choices=["pending", "in_progress", "blocked", "done", "failed"])
    p_task_upd.add_argument("--next-action")
    p_task_upd.add_argument("--priority", choices=["low", "medium", "high", "critical"])

    # bug
    p_bug = sub.add_parser("bug", help="Manage bugs")
    bug_sub = p_bug.add_subparsers(dest="bug_command")

    p_bug_add = bug_sub.add_parser("add")
    p_bug_add.add_argument("--project", required=True, metavar="NAME_OR_ID")
    p_bug_add.add_argument("--summary", required=True)
    p_bug_add.add_argument("--location")
    p_bug_add.add_argument("--severity", choices=["low", "medium", "high", "critical"], default="medium")

    p_bug_res = bug_sub.add_parser("resolve")
    p_bug_res.add_argument("--project", required=True, metavar="NAME_OR_ID")
    p_bug_res.add_argument("--bug-id", required=True)
    p_bug_res.add_argument("--fix", default="resolved")

    # memory
    p_mem = sub.add_parser("memory", help="Manage memory items")
    mem_sub = p_mem.add_subparsers(dest="memory_command")

    p_mem_add = mem_sub.add_parser("add")
    p_mem_add.add_argument("--project", required=True, metavar="NAME_OR_ID")
    p_mem_add.add_argument("--kind",
        choices=["decision", "bug", "constraint", "insight", "module", "dependency"],
        default="insight")
    p_mem_add.add_argument("--content", required=True)
    p_mem_add.add_argument("--relevance", type=float)
    p_mem_add.add_argument("--confidence", type=float)
    p_mem_add.add_argument("--tags", nargs="*")

    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    dispatch = {
        "list":     cmd_list,
        "init":     cmd_init,
        "status":   cmd_status,
        "show":     cmd_show,
        "compress": cmd_compress,
        "rollback": cmd_rollback,
        "export":   cmd_export,
        "import":   cmd_import,
        "history":  cmd_history,
        "metrics":  cmd_metrics,
    }

    if args.command in dispatch:
        dispatch[args.command](args)
    elif args.command == "task":
        if args.task_command == "add":
            cmd_task_add(args)
        elif args.task_command == "done":
            cmd_task_done(args)
        elif args.task_command == "update":
            cmd_task_update(args)
        else:
            parser.parse_args(["task", "--help"])
    elif args.command == "bug":
        if args.bug_command == "add":
            cmd_bug_add(args)
        elif args.bug_command == "resolve":
            cmd_bug_resolve(args)
        else:
            parser.parse_args(["bug", "--help"])
    elif args.command == "memory":
        if args.memory_command == "add":
            cmd_memory_add(args)
        else:
            parser.parse_args(["memory", "--help"])
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
