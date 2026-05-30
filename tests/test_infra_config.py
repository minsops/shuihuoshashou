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
    assert "healthcheck:" in compose


def test_dockerfile_packages_gateway_app() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "FROM python:3.11-slim" in dockerfile
    assert "pip install --no-cache-dir -e ." in dockerfile
    assert "uvicorn" in dockerfile
    assert "services.gateway.app:app" in dockerfile
