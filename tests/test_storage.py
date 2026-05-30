from __future__ import annotations

from pathlib import Path

from libs.common.config import get_settings
from libs.common.storage import LocalArtifactStore, S3CompatibleArtifactStore, get_artifact_store


def test_default_artifact_store_is_local(monkeypatch) -> None:
    monkeypatch.delenv("OBJECT_STORAGE_ENDPOINT", raising=False)
    get_settings.cache_clear()

    assert isinstance(get_artifact_store(), LocalArtifactStore)


def test_s3_artifact_store_returns_stable_uri(tmp_path: Path) -> None:
    path = tmp_path / "report.pdf"
    path.write_bytes(b"%PDF")
    store = S3CompatibleArtifactStore(endpoint="http://minio:9000", bucket="reports")

    artifact = store.put_file("reports/demo.pdf", path, "application/pdf")

    assert artifact.uri == "s3://reports/reports/demo.pdf"
    assert artifact.path == str(path)
    assert artifact.content_type == "application/pdf"
