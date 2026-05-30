from __future__ import annotations

from datetime import UTC, datetime
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar
from uuid import uuid4

from libs.common.config import get_settings
from libs.common.database import dumps
from libs.common.events import event_bus

T = TypeVar("T")


@dataclass(frozen=True)
class TaskRecord(Generic[T]):
    task_id: str
    name: str
    payload: dict[str, Any]
    status: str
    result: T | None = None
    error: str | None = None


@dataclass
class LocalTaskQueue:
    _history: list[TaskRecord[Any]] = field(default_factory=list)

    def enqueue(
        self,
        name: str,
        payload: dict[str, Any],
        handler: Callable[[dict[str, Any]], T],
    ) -> TaskRecord[T]:
        task_id = str(uuid4())
        event_bus.publish_nowait("task.enqueued", {"task_id": task_id, "name": name, "payload": payload})
        try:
            result = handler(payload)
        except Exception as exc:
            record: TaskRecord[T] = TaskRecord(
                task_id=task_id,
                name=name,
                payload=payload,
                status="failed",
                error=str(exc),
            )
            self._history.append(record)
            event_bus.publish_nowait(
                "task.failed",
                {"task_id": task_id, "name": name, "error": str(exc)},
            )
            raise

        record = TaskRecord(
            task_id=task_id,
            name=name,
            payload=payload,
            status="completed",
            result=result,
        )
        self._history.append(record)
        event_bus.publish_nowait("task.completed", {"task_id": task_id, "name": name})
        return record

    def history(self, name: str | None = None) -> list[TaskRecord[Any]]:
        if name is None:
            return list(self._history)
        return [record for record in self._history if record.name == name]

    def reset(self) -> None:
        self._history.clear()


class RedisStreamPublisher:
    def __init__(self, redis_url: str, stream_prefix: str = "shuihuo", client: Any | None = None) -> None:
        self.redis_url = redis_url
        self.stream_prefix = stream_prefix.strip(":")
        self.client = client or self._connect(redis_url)

    @staticmethod
    def _connect(redis_url: str) -> Any:
        try:
            from redis import Redis
        except ImportError as exc:
            raise RuntimeError(
                "Redis task backend requires optional redis dependencies. "
                "Install with `pip install -e .[redis]`."
            ) from exc
        return Redis.from_url(redis_url, decode_responses=True)

    def publish_task(self, task_id: str, name: str, payload: dict[str, Any]) -> str:
        stream = f"{self.stream_prefix}:tasks:{name}"
        message = {
            "task_id": task_id,
            "name": name,
            "payload": dumps(payload),
            "created_at": datetime.now(UTC).isoformat(),
        }
        return str(self.client.xadd(stream, message))


@dataclass
class RedisBackedTaskQueue(LocalTaskQueue):
    publisher: RedisStreamPublisher | None = None

    def __post_init__(self) -> None:
        if self.publisher is None:
            settings = get_settings()
            self.publisher = RedisStreamPublisher(
                redis_url=settings.redis_url,
                stream_prefix=settings.redis_stream_prefix,
            )

    def enqueue(
        self,
        name: str,
        payload: dict[str, Any],
        handler: Callable[[dict[str, Any]], T],
    ) -> TaskRecord[T]:
        task_id = str(uuid4())
        stream_id = self.publisher.publish_task(task_id, name, payload) if self.publisher else ""
        event_bus.publish_nowait(
            "task.enqueued",
            {"task_id": task_id, "name": name, "payload": payload, "stream_id": stream_id},
        )
        try:
            result = handler(payload)
        except Exception as exc:
            record: TaskRecord[T] = TaskRecord(
                task_id=task_id,
                name=name,
                payload=payload,
                status="failed",
                error=str(exc),
            )
            self._history.append(record)
            event_bus.publish_nowait(
                "task.failed",
                {"task_id": task_id, "name": name, "error": str(exc), "stream_id": stream_id},
            )
            raise

        record = TaskRecord(
            task_id=task_id,
            name=name,
            payload=payload,
            status="completed",
            result=result,
        )
        self._history.append(record)
        event_bus.publish_nowait(
            "task.completed",
            {"task_id": task_id, "name": name, "stream_id": stream_id},
        )
        return record


def get_task_queue() -> LocalTaskQueue:
    settings = get_settings()
    if settings.offline_task_backend == "redis_stream":
        return RedisBackedTaskQueue()
    return LocalTaskQueue()


task_queue = get_task_queue()
