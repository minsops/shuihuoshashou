from __future__ import annotations

import builtins

import pytest

from libs.common.config import get_settings
from libs.common.database import (
    _translate_postgres_query,
    connect,
    get_database_target,
    init_db,
    loads,
)


def test_database_target_detects_sqlite() -> None:
    target = get_database_target("sqlite:///data/demo.db")

    assert target.dialect == "sqlite"
    assert str(target.path) == "data/demo.db"


def test_database_target_detects_postgres() -> None:
    target = get_database_target("postgresql://user:pass@localhost:5432/app")

    assert target.dialect == "postgresql"
    assert target.path is None


def test_database_target_rejects_unknown_dialect() -> None:
    with pytest.raises(ValueError, match="unsupported DATABASE_URL dialect"):
        get_database_target("mysql://localhost/app")


def test_runtime_postgres_connection_requires_optional_dependency(monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/app")
    get_settings.cache_clear()
    real_import = builtins.__import__

    def fake_import(name, globals_=None, locals_=None, fromlist=(), level=0):
        if name == "psycopg" or name.startswith("psycopg."):
            raise ImportError("blocked in test")
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="optional postgres dependencies"):
        with connect():
            pass


def test_postgres_query_translation_handles_params_and_upserts() -> None:
    query = """
        INSERT OR REPLACE INTO scores
        (interview_id, dimensions, total_score, risk_notes, recommendation, payload)
        VALUES (?, ?, ?, ?, ?, ?)
    """

    translated = _translate_postgres_query(query)

    assert "INSERT INTO scores" in translated
    assert translated.count("%s") == 6
    assert "ON CONFLICT (interview_id) DO UPDATE SET" in translated
    assert "payload = EXCLUDED.payload" in translated


def test_loads_accepts_postgres_json_values() -> None:
    payload = {"score": 88}

    assert loads(payload) == payload


def test_sqlite_init_migrates_existing_score_columns(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'old.db'}")
    get_settings.cache_clear()
    with connect() as conn:
        conn.executescript(
            """
            CREATE TABLE scores (
                interview_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            );
            CREATE TABLE aigc_results (
                id TEXT PRIMARY KEY,
                interview_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            """
        )

    init_db()

    with connect() as conn:
        score_columns = {row["name"] for row in conn.execute("PRAGMA table_info(scores)").fetchall()}
        aigc_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(aigc_results)").fetchall()
        }

    assert {"dimensions", "total_score", "risk_notes", "recommendation"} <= score_columns
    assert {"ai_generated_prob", "template_similarity", "matched_template", "flagged"} <= aigc_columns
