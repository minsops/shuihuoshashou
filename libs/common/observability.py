from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import Lock
from time import monotonic


def _label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


@dataclass
class RequestStats:
    count: int = 0
    duration_sum: float = 0.0


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = Lock()
        self._requests: dict[tuple[str, str, int], RequestStats] = defaultdict(RequestStats)

    def record_request(self, method: str, path: str, status_code: int, duration_seconds: float) -> None:
        key = (method.upper(), path, status_code)
        with self._lock:
            stats = self._requests[key]
            stats.count += 1
            stats.duration_sum += duration_seconds

    def render_prometheus(self) -> str:
        lines = [
            "# HELP shuihuo_http_requests_total Total HTTP requests.",
            "# TYPE shuihuo_http_requests_total counter",
        ]
        with self._lock:
            snapshot = sorted(self._requests.items())
        for (method, path, status), stats in snapshot:
            labels = (
                f'method="{_label_value(method)}",'
                f'path="{_label_value(path)}",'
                f'status="{status}"'
            )
            lines.append(f"shuihuo_http_requests_total{{{labels}}} {stats.count}")

        lines.extend(
            [
                "# HELP shuihuo_http_request_duration_seconds_sum Total HTTP request duration.",
                "# TYPE shuihuo_http_request_duration_seconds_sum counter",
            ]
        )
        for (method, path, status), stats in snapshot:
            labels = (
                f'method="{_label_value(method)}",'
                f'path="{_label_value(path)}",'
                f'status="{status}"'
            )
            lines.append(
                f"shuihuo_http_request_duration_seconds_sum{{{labels}}} "
                f"{stats.duration_sum:.6f}"
            )

        lines.extend(
            [
                "# HELP shuihuo_http_request_duration_seconds_count Counted HTTP request durations.",
                "# TYPE shuihuo_http_request_duration_seconds_count counter",
            ]
        )
        for (method, path, status), stats in snapshot:
            labels = (
                f'method="{_label_value(method)}",'
                f'path="{_label_value(path)}",'
                f'status="{status}"'
            )
            lines.append(f"shuihuo_http_request_duration_seconds_count{{{labels}}} {stats.count}")
        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        with self._lock:
            self._requests.clear()


@dataclass
class RateLimitDecision:
    allowed: bool
    retry_after_seconds: int = 0


@dataclass
class SlidingWindowRateLimiter:
    requests_per_minute: int
    window_seconds: int = 60
    _events: dict[str, deque[float]] = field(default_factory=lambda: defaultdict(deque))
    _lock: Lock = field(default_factory=Lock)

    def check(self, key: str, now: float | None = None) -> RateLimitDecision:
        if self.requests_per_minute <= 0:
            return RateLimitDecision(allowed=True)

        current = monotonic() if now is None else now
        cutoff = current - self.window_seconds
        with self._lock:
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.requests_per_minute:
                retry_after = max(1, int(self.window_seconds - (current - events[0])))
                return RateLimitDecision(allowed=False, retry_after_seconds=retry_after)
            events.append(current)
        return RateLimitDecision(allowed=True)

    def reset(self) -> None:
        with self._lock:
            self._events.clear()


metrics_registry = MetricsRegistry()
rate_limiter = SlidingWindowRateLimiter(requests_per_minute=120)
