"""
PRM Injection — State → Prompt Block
Converts a PRMState into a compact, structured text block that can be
prepended to any model prompt.

The output is deliberately terse: the model should read it like a developer
reading their own notes — not like a novel.

Supported formats:
  - TEXT  (default): human-readable, Markdown-lite
  - JSON  : raw structured dict (for models that handle JSON well)
  - XML   : for models expecting XML tags (e.g. Anthropic system prompt style)
"""

from __future__ import annotations

import json
from enum import Enum
from typing import List, Optional

from .schema import PRMState, TaskStatus


class InjectionFormat(str, Enum):
    TEXT = "text"
    JSON = "json"
    XML = "xml"


# Maximum characters for the injected block (safety cap)
_MAX_CHARS = 6_000


def build_prompt_block(
    state: PRMState,
    fmt: InjectionFormat = InjectionFormat.TEXT,
    include_bugs: bool = True,
    include_memory: bool = True,
    include_tasks: bool = True,
    max_chars: int = _MAX_CHARS,
) -> str:
    """
    Convert PRMState → prompt-injectable string.

    Args:
        state:          The current PRM state.
        fmt:            Output format (TEXT / JSON / XML).
        include_bugs:   Include unresolved bug list.
        include_memory: Include memory items.
        include_tasks:  Include task graph summary.
        max_chars:      Hard character cap; truncates gracefully.

    Returns:
        A string ready to be prepended to a system or user prompt.
    """
    if fmt == InjectionFormat.JSON:
        block = _build_json(state, include_bugs, include_memory, include_tasks)
    elif fmt == InjectionFormat.XML:
        block = _build_xml(state, include_bugs, include_memory, include_tasks)
    else:
        block = _build_text(state, include_bugs, include_memory, include_tasks)

    if len(block) > max_chars:
        truncation_msg = "\n[PRM TRUNCATED — run compression to reduce state size]"
        block = block[: max_chars - len(truncation_msg)] + truncation_msg

    return block


# ─────────────────────────────────────────────
# Format builders
# ─────────────────────────────────────────────


def _build_text(
    state: PRMState,
    include_bugs: bool,
    include_memory: bool,
    include_tasks: bool,
) -> str:
    lines: List[str] = []
    lines.append("═══════════════════════════════════════")
    lines.append("  PROJECT STATE  (PRM v{})".format(state.version))
    lines.append("═══════════════════════════════════════")
    lines.append(f"Project : {state.project_name}")
    lines.append(f"Goal    : {state.goal or '(not set)'}")
    lines.append(f"Stage   : {state.current_stage or '(not set)'}")
    if state.constraints:
        lines.append("Constraints:")
        for c in state.constraints:
            lines.append(f"  • {c}")

    if include_tasks:
        active = state.active_tasks()
        if active:
            lines.append("\n── Active Tasks ────────────────────────")
            for t in sorted(active, key=lambda x: x.priority, reverse=True):
                blocked = ""
                if t.depends_on:
                    blocked = "  [BLOCKED]" if t.is_blocked(state.tasks) else ""
                lines.append(f"  [{t.priority.upper()}] {t.title}  ({t.status}){blocked}")
                if t.next_action:
                    lines.append(f"    → {t.next_action}")

    if include_bugs:
        unresolved = state.unresolved_bugs()
        if unresolved:
            lines.append("\n── Known Bugs (unresolved) ─────────────")
            for b in unresolved:
                loc = f"  @ {b.location}" if b.location else ""
                lines.append(f"  [{b.severity.upper()}] {b.summary}{loc}")

    if include_memory:
        if state.memory:
            lines.append("\n── Memory Items ────────────────────────")
            # Sort by relevance descending, show top items
            top = sorted(state.memory, key=lambda m: m.relevance, reverse=True)[:20]
            for item in top:
                score = f"rel={item.relevance:.2f} conf={item.confidence:.2f}"
                lines.append(f"  [{item.kind.upper()}] ({score}) {item.content}")

    next_actions = state.next_actions()
    if next_actions:
        lines.append("\n── Next Actions ────────────────────────")
        for a in next_actions[:5]:
            lines.append(f"  {a}")

    lines.append("═══════════════════════════════════════\n")
    return "\n".join(lines)


def _build_json(
    state: PRMState,
    include_bugs: bool,
    include_memory: bool,
    include_tasks: bool,
) -> str:
    d: dict = {
        "prm_version": state.version,
        "project": state.project_name,
        "goal": state.goal,
        "stage": state.current_stage,
        "constraints": state.constraints,
        "goal_confidence": state.goal_confidence,
    }
    if include_tasks:
        d["tasks"] = [
            {
                "id": t.id,
                "title": t.title,
                "status": t.status,
                "priority": t.priority,
                "next_action": t.next_action,
                "depends_on": t.depends_on,
            }
            for t in state.active_tasks()
        ]
    if include_bugs:
        d["unresolved_bugs"] = [
            {"id": b.id, "summary": b.summary, "severity": b.severity, "location": b.location}
            for b in state.unresolved_bugs()
        ]
    if include_memory:
        top = sorted(state.memory, key=lambda m: m.relevance, reverse=True)[:20]
        d["memory"] = [
            {"kind": m.kind, "content": m.content, "relevance": m.relevance, "confidence": m.confidence}
            for m in top
        ]
    return json.dumps(d, indent=2)


def _build_xml(
    state: PRMState,
    include_bugs: bool,
    include_memory: bool,
    include_tasks: bool,
) -> str:
    def esc(s: str) -> str:
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    lines: List[str] = ["<project_state>"]
    lines.append(f"  <version>{state.version}</version>")
    lines.append(f"  <project_name>{esc(state.project_name)}</project_name>")
    lines.append(f"  <goal>{esc(state.goal)}</goal>")
    lines.append(f"  <stage>{esc(state.current_stage)}</stage>")

    if state.constraints:
        lines.append("  <constraints>")
        for c in state.constraints:
            lines.append(f"    <item>{esc(c)}</item>")
        lines.append("  </constraints>")

    if include_tasks:
        lines.append("  <tasks>")
        for t in state.active_tasks():
            lines.append(f'    <task id="{t.id}" priority="{t.priority}" status="{t.status}">')
            lines.append(f"      <title>{esc(t.title)}</title>")
            if t.next_action:
                lines.append(f"      <next_action>{esc(t.next_action)}</next_action>")
            lines.append("    </task>")
        lines.append("  </tasks>")

    if include_bugs:
        lines.append("  <bugs>")
        for b in state.unresolved_bugs():
            lines.append(f'    <bug id="{b.id}" severity="{b.severity}">')
            lines.append(f"      <summary>{esc(b.summary)}</summary>")
            if b.location:
                lines.append(f"      <location>{esc(b.location)}</location>")
            lines.append("    </bug>")
        lines.append("  </bugs>")

    if include_memory:
        top = sorted(state.memory, key=lambda m: m.relevance, reverse=True)[:20]
        lines.append("  <memory>")
        for m in top:
            lines.append(
                f'    <item kind="{m.kind}" relevance="{m.relevance}" confidence="{m.confidence}">'
                f"{esc(m.content)}</item>"
            )
        lines.append("  </memory>")

    lines.append("</project_state>")
    return "\n".join(lines)
