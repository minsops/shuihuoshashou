from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from libs.common.config import get_settings


@dataclass(frozen=True)
class StoredArtifact:
    name: str
    path: str
    uri: str
    content_type: str


class ArtifactStore:
    def put_file(self, name: str, path: Path, content_type: str) -> StoredArtifact:
        raise NotImplementedError


class LocalArtifactStore(ArtifactStore):
    def put_file(self, name: str, path: Path, content_type: str) -> StoredArtifact:
        return StoredArtifact(
            name=name,
            path=str(path),
            uri=path.resolve().as_uri(),
            content_type=content_type,
        )


class S3CompatibleArtifactStore(ArtifactStore):
    def __init__(self, endpoint: str, bucket: str) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.bucket = bucket

    def put_file(self, name: str, path: Path, content_type: str) -> StoredArtifact:
        # Upload is intentionally deferred until a concrete S3 client is configured.
        # The URI is stable and lets report metadata carry the deployment target.
        uri = f"s3://{self.bucket}/{name}"
        return StoredArtifact(name=name, path=str(path), uri=uri, content_type=content_type)


def get_artifact_store() -> ArtifactStore:
    settings = get_settings()
    if settings.object_storage_endpoint:
        return S3CompatibleArtifactStore(
            endpoint=settings.object_storage_endpoint,
            bucket=settings.object_storage_bucket,
        )
    return LocalArtifactStore()
