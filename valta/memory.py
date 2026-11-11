from __future__ import annotations

import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from valta.config import Settings


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Record:
    id: str
    kind: str
    title: str
    text: str
    url: str | None
    source: str | None
    created_at: str


class Memory:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(settings.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init()

    def _init(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS records (
              id TEXT PRIMARY KEY,
              kind TEXT NOT NULL,
              title TEXT,
              text TEXT NOT NULL,
              url TEXT,
              source TEXT,
              created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS goals (
              id TEXT PRIMARY KEY,
              text TEXT NOT NULL,
              status TEXT NOT NULL,
              created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS projects (
              id TEXT PRIMARY KEY,
              name TEXT NOT NULL,
              path TEXT NOT NULL,
              created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS agents (
              id TEXT PRIMARY KEY,
              name TEXT NOT NULL,
              role TEXT NOT NULL,
              model TEXT NOT NULL,
              tools TEXT NOT NULL,
              memory_scope TEXT NOT NULL,
              created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS training_examples (
              id TEXT PRIMARY KEY,
              user_text TEXT NOT NULL,
              assistant_text TEXT NOT NULL,
              source_id TEXT,
              created_at TEXT NOT NULL,
              used INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS topics (
              id TEXT PRIMARY KEY,
              topic TEXT NOT NULL,
              status TEXT NOT NULL,
              source TEXT,
              created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_records_kind ON records(kind, created_at);
            """
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def add_record(
        self,
        text: str,
        kind: str,
        title: str = "",
        url: str | None = None,
        source: str | None = None,
        record_id: str | None = None,
    ) -> Record:
        record = Record(
            id=record_id or uuid.uuid4().hex,
            kind=kind,
            title=title,
            text=text,
            url=url,
            source=source,
            created_at=_now(),
        )
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO records (id, kind, title, text, url, source, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (record.id, record.kind, record.title, record.text, record.url, record.source, record.created_at),
            )
            self.conn.commit()
        return record

    def get(self, record_id: str) -> Record | None:
        row = self.conn.execute("SELECT * FROM records WHERE id = ?", (record_id,)).fetchone()
        return _record(row) if row else None

    def get_many(self, ids: list[str]) -> list[Record]:
        found = []
        for record_id in ids:
            record = self.get(record_id)
            if record:
                found.append(record)
        return found

    def recent(self, limit: int = 10, kind: str | None = None) -> list[Record]:
        if kind:
            rows = self.conn.execute(
                "SELECT * FROM records WHERE kind = ? ORDER BY created_at DESC LIMIT ?",
                (kind, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM records ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_record(row) for row in rows]

    def search_text(self, query: str, limit: int = 20) -> list[Record]:
        like = f"%{query}%"
        rows = self.conn.execute(
            """
            SELECT * FROM records
            WHERE text LIKE ? OR title LIKE ? OR IFNULL(url, '') LIKE ?
            ORDER BY created_at DESC LIMIT ?
            """,
            (like, like, like, limit),
        ).fetchall()
        return [_record(row) for row in rows]

    def find_ids(self, prefix: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT id FROM records WHERE id = ? OR id LIKE ?",
            (prefix, prefix + "%"),
        ).fetchall()
        return [row["id"] for row in rows]

    def delete_record(self, record_id: str) -> bool:
        with self._lock:
            cursor = self.conn.execute("DELETE FROM records WHERE id = ?", (record_id,))
            self.conn.commit()
        return cursor.rowcount > 0

    def source_exists(self, source: str) -> bool:
        row = self.conn.execute("SELECT 1 FROM records WHERE source = ? LIMIT 1", (source,)).fetchone()
        return row is not None

    def add_goal(self, text: str) -> str:
        goal_id = uuid.uuid4().hex
        with self._lock:
            self.conn.execute(
                "INSERT INTO goals (id, text, status, created_at) VALUES (?, ?, ?, ?)",
                (goal_id, text, "open", _now()),
            )
            self.conn.commit()
        return goal_id

    def ensure_project(self, path: Path) -> str:
        resolved = str(path.resolve())
        row = self.conn.execute("SELECT id FROM projects WHERE path = ?", (resolved,)).fetchone()
        if row:
            return row["id"]
        project_id = uuid.uuid4().hex
        with self._lock:
            self.conn.execute(
                "INSERT INTO projects (id, name, path, created_at) VALUES (?, ?, ?, ?)",
                (project_id, path.name or resolved, resolved, _now()),
            )
            self.conn.commit()
        return project_id

    def ensure_default_agent(self) -> str:
        row = self.conn.execute("SELECT id FROM agents WHERE name = ?", ("valta",)).fetchone()
        if row:
            return row["id"]
        agent_id = uuid.uuid4().hex
        tools = "search_memory,save_note,read_file,list_directory"
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO agents (id, name, role, model, tools, memory_scope, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    agent_id,
                    "valta",
                    "companion",
                    self.settings.teacher_model,
                    tools,
                    "all",
                    _now(),
                ),
            )
            self.conn.commit()
        return agent_id

    def ensure_crew(self) -> None:
        crew = (
            ("planner", "Break a goal into a short plan. Do not edit files.", "read_file,list_directory,search_memory"),
            ("coder", "Edit files inside the project to carry out the plan.", "read_file,list_directory,write_file,search_memory,run_tests"),
            ("tester", "Run the tests and report the result exactly.", "run_tests"),
            ("reviewer", "Say whether the work meets the goal after the tests.", "read_file,list_directory,search_memory"),
        )
        for name, role, tools in crew:
            row = self.conn.execute("SELECT id FROM agents WHERE name = ?", (name,)).fetchone()
            if row:
                continue
            with self._lock:
                self.conn.execute(
                    """
                    INSERT INTO agents (id, name, role, model, tools, memory_scope, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (uuid.uuid4().hex, name, role, self.settings.teacher_model, tools, "project", _now()),
                )
                self.conn.commit()

    def add_example(self, user_text: str, assistant_text: str, source_id: str | None = None) -> str:
        example_id = uuid.uuid4().hex
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO training_examples (id, user_text, assistant_text, source_id, created_at, used)
                VALUES (?, ?, ?, ?, ?, 0)
                """,
                (example_id, user_text, assistant_text, source_id, _now()),
            )
            self.conn.commit()
        return example_id

    def all_examples(self) -> list[sqlite3.Row]:
        """Every example, used or not. Each fine-tune starts from the base model, so it must see them all."""
        return self.conn.execute("SELECT * FROM training_examples ORDER BY created_at").fetchall()

    def example_exists(self, user_text: str, assistant_text: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM training_examples WHERE user_text = ? AND assistant_text = ? LIMIT 1",
            (user_text, assistant_text),
        ).fetchone()
        return row is not None

    def unused_examples(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM training_examples WHERE used = 0 ORDER BY created_at"
        ).fetchall()

    def example_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM training_examples WHERE used = 0").fetchone()
        return int(row["n"])

    def mark_examples_used(self, ids: list[str]) -> None:
        with self._lock:
            self.conn.executemany(
                "UPDATE training_examples SET used = 1 WHERE id = ?",
                [(example_id,) for example_id in ids],
            )
            self.conn.commit()

    def enqueue_topic(self, topic: str, source: str = "") -> None:
        cleaned = " ".join(topic.split())
        if not cleaned:
            return
        existing = self.conn.execute(
            "SELECT 1 FROM topics WHERE topic = ? AND status = 'pending' LIMIT 1",
            (cleaned,),
        ).fetchone()
        if existing:
            return
        with self._lock:
            self.conn.execute(
                "INSERT INTO topics (id, topic, status, source, created_at) VALUES (?, ?, ?, ?, ?)",
                (uuid.uuid4().hex, cleaned, "pending", source, _now()),
            )
            self.conn.commit()

    def next_topic(self) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM topics WHERE status = 'pending' ORDER BY created_at LIMIT 1"
        ).fetchone()

    def finish_topic(self, topic_id: str) -> None:
        with self._lock:
            self.conn.execute("UPDATE topics SET status = 'done' WHERE id = ?", (topic_id,))
            self.conn.commit()

    def headings(self, limit: int = 20) -> list[str]:
        rows = self.conn.execute(
            """
            SELECT title FROM records
            WHERE kind = 'book' AND title != ''
            ORDER BY created_at DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [row["title"] for row in rows]


def _record(row: sqlite3.Row) -> Record:
    return Record(
        id=row["id"],
        kind=row["kind"],
        title=row["title"] or "",
        text=row["text"],
        url=row["url"],
        source=row["source"],
        created_at=row["created_at"],
    )
