from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

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


_POSTGRES_UPSERT_KEYS = {
    "qa_turns": "id",
    "scores": "interview_id",
    "aigc_results": "id",
    "reports": "interview_id",
}

_POSTGRES_JSON_COLUMNS = {
    ("jobs", "competency_model"),
    ("interviews", "context"),
    ("qa_turns", "payload"),
    ("scores", "dimensions"),
    ("scores", "risk_notes"),
    ("scores", "payload"),
    ("aigc_results", "payload"),
    ("probe_patterns", "embedding"),
    ("reports", "payload"),
}


def _translate_postgres_query(query: str) -> str:
    query = _translate_sqlite_upsert(query)
    return query.replace("?", "%s")


def _translate_sqlite_upsert(query: str) -> str:
    match = re.search(
        r"INSERT\s+OR\s+REPLACE\s+INTO\s+(\w+)\s*\(([^)]+)\)\s*VALUES\s*\(([^)]+)\)",
        query,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return query
    table = match.group(1)
    conflict_column = _POSTGRES_UPSERT_KEYS.get(table)
    if not conflict_column:
        return query
    columns = [column.strip() for column in match.group(2).split(",")]
    assignments = [
        f"{column} = EXCLUDED.{column}" for column in columns if column != conflict_column
    ]
    if not assignments:
        conflict = f"ON CONFLICT ({conflict_column}) DO NOTHING"
    else:
        conflict = f"ON CONFLICT ({conflict_column}) DO UPDATE SET {', '.join(assignments)}"
    return f"INSERT INTO {table} ({match.group(2)}) VALUES ({match.group(3)}) {conflict}"


def _json_columns_for_insert(query: str) -> list[tuple[int, str]]:
    match = re.search(
        r"INSERT(?:\s+OR\s+REPLACE)?\s+INTO\s+(\w+)\s*\(([^)]+)\)",
        query,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return []
    table = match.group(1)
    columns = [column.strip() for column in match.group(2).split(",")]
    return [
        (index, column)
        for index, column in enumerate(columns)
        if (table, column) in _POSTGRES_JSON_COLUMNS
    ]


def _json_param_indexes_for_update(query: str) -> list[int]:
    match = re.search(r"UPDATE\s+(\w+)\s+SET\s+(.+?)\s+WHERE\s+", query, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return []
    table = match.group(1)
    assignments = [assignment.strip() for assignment in match.group(2).split(",")]
    indexes: list[int] = []
    param_index = 0
    for assignment in assignments:
        column = assignment.split("=", 1)[0].strip()
        if "?" in assignment or "%s" in assignment:
            if (table, column) in _POSTGRES_JSON_COLUMNS:
                indexes.append(param_index)
            param_index += assignment.count("?") + assignment.count("%s")
    return indexes


def _adapt_postgres_params(query: str, params: Sequence[Any] | None) -> Sequence[Any] | None:
    if params is None:
        return None
    indexes = {index for index, _ in _json_columns_for_insert(query)}
    indexes.update(_json_param_indexes_for_update(query))
    if not indexes:
        return params
    from psycopg.types.json import Jsonb

    adapted: list[Any] = list(params)
    for index in indexes:
        if index >= len(adapted):
            continue
        value = adapted[index]
        if isinstance(value, str):
            value = loads(value)
        adapted[index] = Jsonb(value)
    return tuple(adapted)


class PostgresConnection:
    def __init__(self, raw: Any) -> None:
        self.raw = raw

    def execute(self, query: str, params: Sequence[Any] | None = None) -> Any:
        translated = _translate_postgres_query(query)
        adapted = _adapt_postgres_params(query, params)
        return self.raw.execute(translated, adapted)

    def executescript(self, script: str) -> None:
        for statement in script.split(";"):
            if statement.strip():
                self.raw.execute(statement)

    def commit(self) -> None:
        self.raw.commit()

    def close(self) -> None:
        self.raw.close()


@contextmanager
def connect() -> Iterator[Any]:
    target = get_database_target()
    if target.dialect == "postgresql":
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError(
                "PostgreSQL DATABASE_URL requires the optional postgres dependencies. "
                "Install with `pip install -e .[postgres]`."
            ) from exc
        conn = PostgresConnection(psycopg.connect(target.url, row_factory=dict_row))
    else:
        conn = sqlite3.connect(_sqlite_path())
        conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    target = get_database_target()
    with connect() as conn:
        if target.dialect == "postgresql":
            schema = Path("db/postgres/001_core_schema.sql").read_text(encoding="utf-8")
            conn.executescript(schema)
            return
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
            CREATE TABLE IF NOT EXISTS probe_patterns (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                competency TEXT NOT NULL,
                pattern TEXT NOT NULL,
                embedding TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS scores (
                interview_id TEXT PRIMARY KEY,
                dimensions TEXT,
                total_score REAL,
                risk_notes TEXT,
                recommendation TEXT,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS aigc_results (
                id TEXT PRIMARY KEY,
                interview_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                ai_generated_prob REAL,
                template_similarity REAL,
                matched_template TEXT,
                flagged INTEGER,
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
        _ensure_sqlite_columns(conn)


def _ensure_sqlite_columns(conn: sqlite3.Connection) -> None:
    existing_scores = {row["name"] for row in conn.execute("PRAGMA table_info(scores)").fetchall()}
    score_columns = {
        "dimensions": "TEXT",
        "total_score": "REAL",
        "risk_notes": "TEXT",
        "recommendation": "TEXT",
    }
    for column, column_type in score_columns.items():
        if column not in existing_scores:
            conn.execute(f"ALTER TABLE scores ADD COLUMN {column} {column_type}")

    existing_aigc = {row["name"] for row in conn.execute("PRAGMA table_info(aigc_results)").fetchall()}
    aigc_columns = {
        "ai_generated_prob": "REAL",
        "template_similarity": "REAL",
        "matched_template": "TEXT",
        "flagged": "INTEGER",
    }
    for column, column_type in aigc_columns.items():
        if column not in existing_aigc:
            conn.execute(f"ALTER TABLE aigc_results ADD COLUMN {column} {column_type}")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS probe_patterns (
            id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            competency TEXT NOT NULL,
            pattern TEXT NOT NULL,
            embedding TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def loads(value: str) -> Any:
    if not isinstance(value, str):
        return value
    return json.loads(value)
