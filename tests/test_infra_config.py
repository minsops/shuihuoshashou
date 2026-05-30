from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_docker_compose_declares_required_infrastructure() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    for service in ["gateway:", "postgres:", "redis:", "minio:"]:
        assert service in compose
    assert "postgres:16-alpine" in compose
    assert "redis:7-alpine" in compose
    assert "minio/minio" in compose
    assert "REDIS_URL: redis://redis:6379/0" in compose
    assert "OBJECT_STORAGE_ENDPOINT: http://minio:9000" in compose
    assert "./db/postgres:/docker-entrypoint-initdb.d:ro" in compose
    assert "healthcheck:" in compose


def test_dockerfile_packages_gateway_app() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "FROM python:3.11-slim" in dockerfile
    assert "pip install --no-cache-dir -e ." in dockerfile
    assert "uvicorn" in dockerfile
    assert "services.gateway.app:app" in dockerfile


def test_env_example_lists_runtime_integration_knobs() -> None:
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")

    for key in [
        "ASR_PROVIDER=",
        "ASR_BASE_URL=",
        "GATEWAY_API_KEY=",
        "SPEAKER_DIARIZATION_PROVIDER=",
        "SPEAKER_DIARIZATION_BASE_URL=",
        "OFFLINE_TASK_EXECUTION=",
        "REDIS_STREAM_PREFIX=",
        "JD_VECTOR_BACKEND=",
        "OBJECT_STORAGE_ACCESS_KEY=",
        "REPORT_DIR=",
    ]:
        assert key in env_example


def test_postgres_schema_matches_core_spec_tables() -> None:
    schema = (ROOT / "db" / "postgres" / "001_core_schema.sql").read_text(encoding="utf-8")

    required_tables = [
        "jobs",
        "candidates",
        "interviews",
        "qa_turns",
        "scores",
        "aigc_results",
        "reports",
        "consents",
    ]
    for table in required_tables:
        assert f"CREATE TABLE IF NOT EXISTS {table}" in schema
    for column in [
        "competency_model JSONB NOT NULL",
        "context JSONB NOT NULL",
        "dimensions JSONB NOT NULL",
        "ai_generated_prob REAL NOT NULL",
        "template_similarity REAL NOT NULL",
        "signal_enabled BOOLEAN NOT NULL DEFAULT false",
    ]:
        assert column in schema
    assert "REFERENCES jobs(id)" in schema
    assert "REFERENCES candidates(id)" in schema
    assert "REFERENCES interviews(id)" in schema
