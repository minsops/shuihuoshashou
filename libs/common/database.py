from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from libs.common.config import get_settings


@dataclass(frozen=True)
class DatabaseTarget:
    dialect: str
    url: str
    path: Path | None = None


def get_database_target(url: str | None = None) -> DatabaseTarget:
    resolved = get_settings().database_url if url is None else url
    if resolved.startswith("sqlite:///"):
        return DatabaseTarget(
            dialect="sqlite",
            url=resolved,
            path=Path(resolved.replace("sqlite:///", "", 1)),
        )
    if resolved.startswith(("postgresql://", "postgres://")):
        return DatabaseTarget(dialect="postgresql", url=resolved)
    raise ValueError(f"unsupported DATABASE_URL dialect: {resolved}")


def _sqlite_path() -> Path:
    target = get_database_target()
    if target.dialect != "sqlite" or target.path is None:
        raise ValueError("runtime persistence currently supports sqlite:/// DATABASE_URL values")
    path = target.path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    target = get_database_target()
    if target.dialect == "postgresql":
        raise NotImplementedError(
            "PostgreSQL schema is available under db/postgres, but runtime data access still "
            "uses SQLite. Set DATABASE_URL=sqlite:///... or implement the PostgreSQL repository."
        )
    conn = sqlite3.connect(_sqlite_path())
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                jd_text TEXT,
                competency_model TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS candidates (
                id TEXT PRIMARY KEY,
                name TEXT,
                resume_text TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS interviews (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                status TEXT NOT NULL,
                context TEXT NOT NULL,
                signal_enabled INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                started_at TEXT,
                ended_at TEXT
            );
            CREATE TABLE IF NOT EXISTS qa_turns (
                id TEXT PRIMARY KEY,
                interview_id TEXT NOT NULL,
                turn_index INTEGER NOT NULL,
                question TEXT NOT NULL,
                question_source TEXT NOT NULL,
                answer TEXT NOT NULL,
                answer_start_ms INTEGER NOT NULL,
                answer_end_ms INTEGER NOT NULL,
                probe_target TEXT,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS scores (
                interview_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS aigc_results (
                id TEXT PRIMARY KEY,
                interview_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reports (
                interview_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                html TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS consents (
                id TEXT PRIMARY KEY,
                candidate_id TEXT NOT NULL,
                consent_type TEXT NOT NULL,
                granted INTEGER NOT NULL,
                granted_at TEXT NOT NULL,
                revoked_at TEXT
            );
            """
        )


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def loads(value: str) -> Any:
    return json.loads(value)
