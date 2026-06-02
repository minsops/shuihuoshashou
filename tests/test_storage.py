from __future__ import annotations

from pathlib import Path

import httpx

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


def test_s3_artifact_store_uploads_when_credentials_are_configured(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200)

    path = tmp_path / "report.html"
    path.write_text("<h1>报告</h1>", encoding="utf-8")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    store = S3CompatibleArtifactStore(
        endpoint="http://minio:9000",
        bucket="reports",
        access_key="minioadmin",
        secret_key="minioadmin",
        region="us-east-1",
        client=client,
    )

    artifact = store.put_file("reports/demo report.html", path, "text/html; charset=utf-8")

    assert artifact.uri == "s3://reports/reports/demo report.html"
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "PUT"
    assert str(request.url) == "http://minio:9000/reports/reports/demo%20report.html"
    assert request.content == "<h1>报告</h1>".encode("utf-8")
    assert request.headers["content-type"] == "text/html; charset=utf-8"
    assert request.headers["x-amz-content-sha256"]
    assert "AWS4-HMAC-SHA256 Credential=minioadmin/" in request.headers["authorization"]


def test_s3_artifact_store_skips_upload_without_credentials(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []
    path = tmp_path / "report.pdf"
    path.write_bytes(b"%PDF")
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: requests.append(request) or httpx.Response(200)
        )
    )
    store = S3CompatibleArtifactStore(
        endpoint="http://minio:9000",
        bucket="reports",
        client=client,
    )

    artifact = store.put_file("reports/demo.pdf", path, "application/pdf")

    assert artifact.uri == "s3://reports/reports/demo.pdf"
    assert requests == []


def test_local_artifact_store_reads_file_uri(tmp_path: Path) -> None:
    path = tmp_path / "report.pdf"
    path.write_bytes(b"%PDF local")
    store = LocalArtifactStore()

    artifact = store.get_file(path.as_uri())

    assert artifact.uri == path.as_uri()
    assert artifact.content == b"%PDF local"
    assert artifact.content_type == "application/octet-stream"


def test_s3_artifact_store_downloads_with_credentials() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=b"%PDF s3", headers={"content-type": "application/pdf"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    store = S3CompatibleArtifactStore(
        endpoint="http://minio:9000",
        bucket="reports",
        access_key="minioadmin",
        secret_key="minioadmin",
        region="us-east-1",
        client=client,
    )

    artifact = store.get_file("s3://reports/reports/demo report.pdf")

    assert artifact.uri == "s3://reports/reports/demo report.pdf"
    assert artifact.content == b"%PDF s3"
    assert artifact.content_type == "application/pdf"
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "GET"
    assert str(request.url) == "http://minio:9000/reports/reports/demo%20report.pdf"
    assert request.headers["x-amz-content-sha256"]
    assert "AWS4-HMAC-SHA256 Credential=minioadmin/" in request.headers["authorization"]
