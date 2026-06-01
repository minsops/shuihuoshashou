from __future__ import annotations

import builtins

from libs.common.config import Settings
from libs.common.observability import RedisFixedWindowRateLimiter, configure_opentelemetry


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, int] = {}
        self.expirations: dict[str, int] = {}

    def incr(self, key: str) -> int:
        self.values[key] = self.values.get(key, 0) + 1
        return self.values[key]

    def expire(self, key: str, seconds: int) -> None:
        self.expirations[key] = seconds

    def ttl(self, key: str) -> int:
        return self.expirations.get(key, 1)


def test_redis_rate_limiter_uses_fixed_window_keys() -> None:
    client = FakeRedis()
    limiter = RedisFixedWindowRateLimiter(
        "redis://localhost:6379/0",
        prefix="test:limit",
        requests_per_minute=2,
        window_seconds=60,
        client=client,
    )

    assert limiter.check("client-1", now=10).allowed is True
    assert limiter.check("client-1", now=20).allowed is True
    limited = limiter.check("client-1", now=30)

    assert limited.allowed is False
    assert limited.retry_after_seconds == 60
    assert client.values == {"test:limit:client-1:0": 3}
    assert client.expirations == {"test:limit:client-1:0": 60}


def test_redis_rate_limiter_starts_new_window() -> None:
    client = FakeRedis()
    limiter = RedisFixedWindowRateLimiter(
        "redis://localhost:6379/0",
        prefix="test:limit",
        requests_per_minute=1,
        window_seconds=60,
        client=client,
    )

    assert limiter.check("client-1", now=10).allowed is True
    assert limiter.check("client-1", now=61).allowed is True

    assert set(client.values) == {"test:limit:client-1:0", "test:limit:client-1:1"}


def test_configure_opentelemetry_skips_without_endpoint() -> None:
    settings = Settings(otel_exporter_otlp_endpoint="")

    result = configure_opentelemetry(object(), settings)

    assert result.enabled is False
    assert result.reason == "endpoint_not_configured"


def test_configure_opentelemetry_skips_when_optional_dependencies_missing(monkeypatch) -> None:
    settings = Settings(otel_exporter_otlp_endpoint="http://collector:4318/v1/traces")
    real_import = builtins.__import__

    def fake_import(name, globals_=None, locals_=None, fromlist=(), level=0):
        if name == "opentelemetry" or name.startswith("opentelemetry."):
            raise ImportError("blocked in test")
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    result = configure_opentelemetry(object(), settings)

    assert result.enabled is False
    assert result.reason == "missing_optional_dependencies"
