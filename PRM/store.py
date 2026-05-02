"""
PRM Store — Persistence Layer
SQLite-backed storage for PRMState with versioning, atomic writes,
and optional JSON file fallback.

Design:
  - Each project maps to one row keyed by project_id
  - Versioned writes — never blindly overwrite
  - Full history log (ring buffer, configurable size)
  - Thread-safe via WAL mode + context manager
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List, Optional

from .schema import PRMState

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS prm_states (
    project_id  TEXT    PRIMARY KEY,
    name        TEXT    NOT NULL,
    version     INTEGER NOT NULL,
    state_json  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS prm_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  TEXT    NOT NULL,
    version     INTEGER NOT NULL,
    state_json  TEXT    NOT NULL,
    saved_at    TEXT    NOT NULL DEFAULT (datetime('now','utc'))
);

CREATE INDEX IF NOT EXISTS idx_history_project ON prm_history(project_id, version);
"""

_HISTORY_LIMIT = 50   # keep last N snapshots per project


class PRMStore:
    """
    Thread-safe SQLite store for PRM states.

    Usage:
        store = PRMStore("./prm.db")
        store.save(state)
        state = store.load("project-id")
    """

    def __init__(self, db_path: str | Path = "./prm.db", history_limit: int = _HISTORY_LIMIT) -> None:
        self.db_path = Path(db_path)
        self.history_limit = history_limit
        self._local = threading.local()
        self._init_db()

    # ── Connection management ────────────────

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    @contextmanager
    def _tx(self) -> Generator[sqlite3.Connection, None, None]:
        conn = self._conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _init_db(self) -> None:
        with self._tx() as conn:
            conn.executescript(_SCHEMA)
        logger.debug("PRMStore initialised at %s", self.db_path)

    # ── Public API ───────────────────────────

    def save(self, state: PRMState) -> None:
        """
        Persist state.  Raises ValueError if a newer version already exists
        (optimistic concurrency — prevents accidental rollbacks).
        """
        payload = state.model_dump_json()
        with self._tx() as conn:
            existing = conn.execute(
                "SELECT version FROM prm_states WHERE project_id = ?", (state.project_id,)
            ).fetchone()

            if existing and existing["version"] > state.version:
                raise ValueError(
                    f"Store has version {existing['version']} but state has version "
                    f"{state.version}. Reload before saving."
                )

            conn.execute(
                """
                INSERT INTO prm_states (project_id, name, version, state_json, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(project_id) DO UPDATE SET
                    name       = excluded.name,
                    version    = excluded.version,
                    state_json = excluded.state_json,
                    updated_at = excluded.updated_at
                """,
                (state.project_id, state.project_name, state.version, payload, state.updated_at),
            )

            # Append to history
            conn.execute(
                "INSERT INTO prm_history (project_id, version, state_json) VALUES (?, ?, ?)",
                (state.project_id, state.version, payload),
            )

            # Prune old history
            conn.execute(
                """
                DELETE FROM prm_history WHERE project_id = ? AND id NOT IN (
                    SELECT id FROM prm_history WHERE project_id = ?
                    ORDER BY version DESC LIMIT ?
                )
                """,
                (state.project_id, state.project_id, self.history_limit),
            )

        logger.info("Saved PRM '%s' version=%d", state.project_name, state.version)

    def load(self, project_id: str) -> Optional[PRMState]:
        """Return the latest state for a project, or None if not found."""
        row = self._conn().execute(
            "SELECT state_json FROM prm_states WHERE project_id = ?", (project_id,)
        ).fetchone()
        if row is None:
            return None
        state = PRMState.model_validate_json(row["state_json"])
        logger.debug("Loaded PRM '%s' version=%d", state.project_name, state.version)
        return state

    def load_by_name(self, name: str) -> Optional[PRMState]:
        """Load a project by its human-readable name (first match)."""
        row = self._conn().execute(
            "SELECT state_json FROM prm_states WHERE name = ? ORDER BY version DESC LIMIT 1",
            (name,),
        ).fetchone()
        if row is None:
            return None
        return PRMState.model_validate_json(row["state_json"])

    def list_projects(self) -> List[dict]:
        rows = self._conn().execute(
            "SELECT project_id, name, version, updated_at FROM prm_states ORDER BY updated_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def load_version(self, project_id: str, version: int) -> Optional[PRMState]:
        """Restore a specific historical version (read-only)."""
        row = self._conn().execute(
            "SELECT state_json FROM prm_history WHERE project_id = ? AND version = ?",
            (project_id, version),
        ).fetchone()
        if row is None:
            return None
        return PRMState.model_validate_json(row["state_json"])

    def history_versions(self, project_id: str) -> List[int]:
        rows = self._conn().execute(
            "SELECT version FROM prm_history WHERE project_id = ? ORDER BY version DESC",
            (project_id,),
        ).fetchall()
        return [r["version"] for r in rows]

    def delete(self, project_id: str) -> None:
        with self._tx() as conn:
            conn.execute("DELETE FROM prm_states WHERE project_id = ?", (project_id,))
            conn.execute("DELETE FROM prm_history WHERE project_id = ?", (project_id,))
        logger.warning("Deleted PRM project_id=%s", project_id)

    def export_json(self, project_id: str, path: str | Path) -> None:
        """Export current state to a plain JSON file (for debugging / backup)."""
        state = self.load(project_id)
        if state is None:
            raise KeyError(f"Project {project_id!r} not found")
        Path(path).write_text(state.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Exported PRM to %s", path)

    def import_json(self, path: str | Path) -> PRMState:
        """Import a JSON file back into the store (merges, does not overwrite newer)."""
        data = Path(path).read_text(encoding="utf-8")
        state = PRMState.model_validate_json(data)
        self.save(state)
        return state

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
