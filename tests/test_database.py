from __future__ import annotations

import pytest

from libs.common.config import get_settings
from libs.common.database import connect, get_database_target


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


def test_runtime_postgres_connection_is_explicitly_not_implemented(monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/app")
    get_settings.cache_clear()

    with pytest.raises(NotImplementedError, match="PostgreSQL schema is available"):
        with connect():
            pass
