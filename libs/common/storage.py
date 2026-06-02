from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

import httpx

from libs.common.config import get_settings


@dataclass(frozen=True)
class StoredArtifact:
    name: str
    path: str
    uri: str
    content_type: str


@dataclass(frozen=True)
class ArtifactContent:
    uri: str
    content: bytes
    content_type: str


class ArtifactStore:
    def artifact_uri(self, name: str, path: Path) -> str:
        raise NotImplementedError

    def put_file(self, name: str, path: Path, content_type: str) -> StoredArtifact:
        raise NotImplementedError

    def get_file(self, uri: str) -> ArtifactContent:
        raise NotImplementedError


class LocalArtifactStore(ArtifactStore):
    def artifact_uri(self, name: str, path: Path) -> str:
        return path.resolve().as_uri()

    def put_file(self, name: str, path: Path, content_type: str) -> StoredArtifact:
        return StoredArtifact(
            name=name,
            path=str(path),
            uri=self.artifact_uri(name, path),
            content_type=content_type,
        )

    def get_file(self, uri: str) -> ArtifactContent:
        parsed = urlparse(uri)
        if parsed.scheme != "file":
            raise ValueError(f"unsupported local artifact uri: {uri}")
        path = Path(unquote(parsed.path))
        return ArtifactContent(
            uri=uri,
            content=path.read_bytes(),
            content_type="application/octet-stream",
        )


class S3CompatibleArtifactStore(ArtifactStore):
    def __init__(
        self,
        endpoint: str,
        bucket: str,
        access_key: str = "",
        secret_key: str = "",
        region: str = "us-east-1",
        client: httpx.Client | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.bucket = bucket
        self.access_key = access_key
        self.secret_key = secret_key
        self.region = region
        self.client = client or httpx.Client(timeout=30)

    def put_file(self, name: str, path: Path, content_type: str) -> StoredArtifact:
        if self.access_key and self.secret_key:
            self._upload_file(name, path, content_type)
        return StoredArtifact(
            name=name,
            path=str(path),
            uri=self.artifact_uri(name, path),
            content_type=content_type,
        )

    def artifact_uri(self, name: str, path: Path) -> str:
        return f"s3://{self.bucket}/{name}"

    def get_file(self, uri: str) -> ArtifactContent:
        key = self._key_from_uri(uri)
        url = self._object_url(key)
        headers = self._signed_headers("GET", url) if self.access_key and self.secret_key else {}
        response = self.client.get(url, headers=headers)
        response.raise_for_status()
        return ArtifactContent(
            uri=uri,
            content=response.content,
            content_type=response.headers.get("content-type", "application/octet-stream"),
        )

    def _upload_file(self, name: str, path: Path, content_type: str) -> None:
        body = path.read_bytes()
        url = self._object_url(name)
        headers = self._signed_headers("PUT", url, body, content_type)
        response = self.client.put(url, content=body, headers=headers)
        response.raise_for_status()

    def _object_url(self, key: str) -> str:
        encoded_key = "/".join(quote(part, safe="") for part in key.split("/"))
        return f"{self.endpoint}/{quote(self.bucket, safe='')}/{encoded_key}"

    def _key_from_uri(self, uri: str) -> str:
        parsed = urlparse(uri)
        if parsed.scheme != "s3" or parsed.netloc != self.bucket:
            raise ValueError(f"unsupported s3 artifact uri: {uri}")
        key = unquote(parsed.path.lstrip("/"))
        if not key:
            raise ValueError(f"missing s3 artifact key: {uri}")
        return key

    def _signed_headers(
        self,
        method: str,
        url: str,
        body: bytes = b"",
        content_type: str | None = None,
    ) -> dict[str, str]:
        now = datetime.now(UTC)
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        date_stamp = now.strftime("%Y%m%d")
        parsed = urlparse(url)
        payload_hash = hashlib.sha256(body).hexdigest()
        headers = {
            "host": parsed.netloc,
            "x-amz-content-sha256": payload_hash,
            "x-amz-date": amz_date,
        }
        if content_type:
            headers["content-type"] = content_type
        signed_headers = ";".join(sorted(headers))
        canonical_headers = "".join(f"{key}:{headers[key]}\n" for key in sorted(headers))
        canonical_request = "\n".join(
            [
                method.upper(),
                parsed.path or "/",
                parsed.query,
                canonical_headers,
                signed_headers,
                payload_hash,
            ]
        )
        credential_scope = f"{date_stamp}/{self.region}/s3/aws4_request"
        string_to_sign = "\n".join(
            [
                "AWS4-HMAC-SHA256",
                amz_date,
                credential_scope,
                hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
            ]
        )
        signing_key = _aws_v4_signing_key(self.secret_key, date_stamp, self.region)
        signature = hmac.new(
            signing_key,
            string_to_sign.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        headers["authorization"] = (
            "AWS4-HMAC-SHA256 "
            f"Credential={self.access_key}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, "
            f"Signature={signature}"
        )
        return headers


def _aws_v4_signing_key(secret_key: str, date_stamp: str, region: str) -> bytes:
    date_key = hmac.new(
        f"AWS4{secret_key}".encode("utf-8"),
        date_stamp.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    region_key = hmac.new(date_key, region.encode("utf-8"), hashlib.sha256).digest()
    service_key = hmac.new(region_key, b"s3", hashlib.sha256).digest()
    return hmac.new(service_key, b"aws4_request", hashlib.sha256).digest()


def get_artifact_store() -> ArtifactStore:
    settings = get_settings()
    if settings.object_storage_endpoint:
        return S3CompatibleArtifactStore(
            endpoint=settings.object_storage_endpoint,
            bucket=settings.object_storage_bucket,
            access_key=settings.object_storage_access_key,
            secret_key=settings.object_storage_secret_key,
            region=settings.object_storage_region,
        )
    return LocalArtifactStore()
