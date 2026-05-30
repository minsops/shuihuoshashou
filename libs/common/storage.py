from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote, urlparse

import httpx

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
        uri = f"s3://{self.bucket}/{name}"
        if self.access_key and self.secret_key:
            self._upload_file(name, path, content_type)
        return StoredArtifact(name=name, path=str(path), uri=uri, content_type=content_type)

    def _upload_file(self, name: str, path: Path, content_type: str) -> None:
        body = path.read_bytes()
        key = "/".join(quote(part, safe="") for part in name.split("/"))
        url = f"{self.endpoint}/{quote(self.bucket, safe='')}/{key}"
        headers = self._signed_put_headers(url, body, content_type)
        response = self.client.put(url, content=body, headers=headers)
        response.raise_for_status()

    def _signed_put_headers(self, url: str, body: bytes, content_type: str) -> dict[str, str]:
        now = datetime.now(UTC)
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        date_stamp = now.strftime("%Y%m%d")
        parsed = urlparse(url)
        payload_hash = hashlib.sha256(body).hexdigest()
        headers = {
            "content-type": content_type,
            "host": parsed.netloc,
            "x-amz-content-sha256": payload_hash,
            "x-amz-date": amz_date,
        }
        signed_headers = ";".join(sorted(headers))
        canonical_headers = "".join(f"{key}:{headers[key]}\n" for key in sorted(headers))
        canonical_request = "\n".join(
            [
                "PUT",
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
