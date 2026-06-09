from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import json
import os
import ssl
import time
import uuid
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable
from urllib.parse import quote, urlencode
from urllib.request import urlopen

DEFAULT_ENDPOINT = "https://nls-meta.cn-shanghai.aliyuncs.com/"
DEFAULT_REGION = "cn-shanghai"
TOKEN_EXPIRY_SAFETY_SECONDS = 300


@dataclass(frozen=True)
class AliyunNLSToken:
    id: str
    expire_time: int | None = None
    user_id: str = ""


class AliyunNLSTokenProvider:
    def __init__(
        self,
        *,
        access_key_id: str,
        access_key_secret: str,
        endpoint: str = DEFAULT_ENDPOINT,
        region_id: str = DEFAULT_REGION,
        create_token_func: Callable[..., AliyunNLSToken] | None = None,
        now_func: Callable[[], float] | None = None,
    ) -> None:
        self.access_key_id = access_key_id
        self.access_key_secret = access_key_secret
        self.endpoint = endpoint
        self.region_id = region_id
        self._create_token = create_token_func or create_token
        self._now = now_func or time.time
        self._cached_token: AliyunNLSToken | None = None
        self._lock = Lock()

    def get_token(self) -> str:
        with self._lock:
            if self._cached_token is not None and self._is_fresh(self._cached_token):
                return self._cached_token.id
            self._cached_token = self._create_token(
                self.access_key_id,
                self.access_key_secret,
                endpoint=self.endpoint,
                region_id=self.region_id,
            )
            return self._cached_token.id

    def invalidate(self) -> None:
        with self._lock:
            self._cached_token = None

    def _is_fresh(self, token: AliyunNLSToken) -> bool:
        if token.expire_time is None:
            return False
        return token.expire_time > int(self._now()) + TOKEN_EXPIRY_SAFETY_SECONDS


def create_token_from_env(*, timeout: float = 10) -> AliyunNLSToken | None:
    access_key_id = os.environ.get("ALIYUN_AK_ID", "").strip()
    access_key_secret = os.environ.get("ALIYUN_AK_SECRET", "").strip()
    if not access_key_id or not access_key_secret:
        return None
    endpoint = os.environ.get("ALIYUN_NLS_TOKEN_ENDPOINT", DEFAULT_ENDPOINT).strip()
    region_id = os.environ.get("ALIYUN_NLS_TOKEN_REGION", DEFAULT_REGION).strip()
    return create_token(
        access_key_id,
        access_key_secret,
        endpoint=endpoint or DEFAULT_ENDPOINT,
        region_id=region_id or DEFAULT_REGION,
        timeout=timeout,
    )


def create_token(
    access_key_id: str,
    access_key_secret: str,
    *,
    endpoint: str = DEFAULT_ENDPOINT,
    region_id: str = DEFAULT_REGION,
    timeout: float = 10,
) -> AliyunNLSToken:
    params = _signed_params(
        access_key_id,
        access_key_secret,
        region_id=region_id,
    )
    query = urlencode(params, quote_via=quote, safe="-_.~")
    url = f"{endpoint.rstrip('/')}?{query}"
    with urlopen(url, timeout=timeout, context=_ssl_context()) as response:
        payload = json.loads(response.read().decode("utf-8"))
    token = _parse_token(payload)
    if token is None:
        err = payload.get("ErrMsg") or payload.get("Message") or payload.get("Code") or payload
        raise RuntimeError(f"Aliyun NLS CreateToken failed: {err}")
    return token


def _signed_params(
    access_key_id: str,
    access_key_secret: str,
    *,
    region_id: str = DEFAULT_REGION,
    timestamp: str | None = None,
    nonce: str | None = None,
) -> dict[str, str]:
    params = {
        "AccessKeyId": access_key_id,
        "Action": "CreateToken",
        "Format": "JSON",
        "RegionId": region_id,
        "SignatureMethod": "HMAC-SHA1",
        "SignatureNonce": nonce or str(uuid.uuid4()),
        "SignatureVersion": "1.0",
        "Timestamp": timestamp or _utc_timestamp(),
        "Version": "2019-02-28",
    }
    canonicalized_query = "&".join(
        f"{_percent_encode(key)}={_percent_encode(value)}"
        for key, value in sorted(params.items())
    )
    string_to_sign = f"GET&{_percent_encode('/')}&{_percent_encode(canonicalized_query)}"
    digest = hmac.new(
        f"{access_key_secret}&".encode("utf-8"),
        string_to_sign.encode("utf-8"),
        hashlib.sha1,
    ).digest()
    params["Signature"] = base64.b64encode(digest).decode("ascii")
    return params


def _parse_token(payload: dict[str, Any]) -> AliyunNLSToken | None:
    raw_token = payload.get("Token")
    if not isinstance(raw_token, dict):
        return None
    token_id = str(raw_token.get("Id") or "").strip()
    if not token_id:
        return None
    expire_time = raw_token.get("ExpireTime")
    return AliyunNLSToken(
        id=token_id,
        expire_time=int(expire_time) if expire_time is not None else None,
        user_id=str(raw_token.get("UserId") or ""),
    )


def _utc_timestamp() -> str:
    now = dt.datetime.now(dt.UTC).replace(microsecond=0)
    return now.strftime("%Y-%m-%dT%H:%M:%SZ")


def _percent_encode(value: str) -> str:
    return quote(str(value), safe="-_.~")


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())
