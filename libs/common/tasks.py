from __future__ import annotations

from datetime import UTC, datetime
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar
from uuid import uuid4

from libs.common.config import get_settings
from libs.common.database import dumps, loads
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

    def enqueue_deferred(self, name: str, payload: dict[str, Any]) -> TaskRecord[Any]:
        task_id = str(uuid4())
        record = TaskRecord(
            task_id=task_id,
            name=name,
            payload=payload,
            status="queued",
        )
        self._history.append(record)
        event_bus.publish_nowait("task.enqueued", {"task_id": task_id, "name": name, "payload": payload})
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
        stream = redis_task_stream(self.stream_prefix, name)
        message = {
            "task_id": task_id,
            "name": name,
            "payload": dumps(payload),
            "created_at": datetime.now(UTC).isoformat(),
        }
        return str(self.client.xadd(stream, message))


def redis_task_stream(stream_prefix: str, task_name: str) -> str:
    return f"{stream_prefix.strip(':')}:tasks:{task_name}"


class RedisStreamWorker:
    def __init__(
        self,
        task_name: str,
        handler: Callable[[dict[str, Any]], Any],
        *,
        redis_url: str,
        stream_prefix: str = "shuihuo",
        group: str = "offline-workers",
        consumer: str = "worker-1",
        client: Any | None = None,
    ) -> None:
        self.task_name = task_name
        self.handler = handler
        self.redis_url = redis_url
        self.stream_prefix = stream_prefix.strip(":")
        self.group = group
        self.consumer = consumer
        self.client = client or RedisStreamPublisher._connect(redis_url)
        self.stream = redis_task_stream(self.stream_prefix, task_name)

    def ensure_group(self) -> None:
        try:
            self.client.xgroup_create(self.stream, self.group, id="0", mkstream=True)
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    def consume_once(self, *, block_ms: int = 1000, count: int = 1) -> int:
        self.ensure_group()
        messages = self.client.xreadgroup(
            self.group,
            self.consumer,
            {self.stream: ">"},
            count=count,
            block=block_ms,
        )
        processed = 0
        for stream, entries in messages:
            for message_id, fields in entries:
                payload = loads(fields["payload"])
                try:
                    self.handler(payload)
                except Exception as exc:
                    event_bus.publish_nowait(
                        "task.worker_failed",
                        {
                            "stream": stream,
                            "message_id": message_id,
                            "task_id": fields.get("task_id", ""),
                            "name": fields.get("name", self.task_name),
                            "error": str(exc),
                        },
                    )
                    raise
                self.client.xack(stream, self.group, message_id)
                event_bus.publish_nowait(
                    "task.worker_completed",
                    {
                        "stream": stream,
                        "message_id": message_id,
                        "task_id": fields.get("task_id", ""),
                        "name": fields.get("name", self.task_name),
                    },
                )
                processed += 1
        return processed


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

    def enqueue_deferred(self, name: str, payload: dict[str, Any]) -> TaskRecord[Any]:
        task_id = str(uuid4())
        stream_id = self.publisher.publish_task(task_id, name, payload) if self.publisher else ""
        record = TaskRecord(
            task_id=task_id,
            name=name,
            payload=payload,
            status="queued",
        )
        self._history.append(record)
        event_bus.publish_nowait(
            "task.enqueued",
            {"task_id": task_id, "name": name, "payload": payload, "stream_id": stream_id},
        )
        return record


class CeleryTaskPublisher:
    def __init__(
        self,
        broker_url: str,
        result_backend: str,
        sender: Any | None = None,
    ) -> None:
        self.broker_url = broker_url
        self.result_backend = result_backend
        self.sender = sender or self._connect(broker_url, result_backend)

    @staticmethod
    def _connect(broker_url: str, result_backend: str) -> Any:
        try:
            from celery import Celery
        except ImportError as exc:
            raise RuntimeError(
                "Celery task backend requires optional celery dependencies. "
                "Install with `pip install -e .[celery]`."
            ) from exc
        return Celery("shuihuo", broker=broker_url, backend=result_backend)

    def publish_task(self, task_id: str, name: str, payload: dict[str, Any]) -> str:
        result = self.sender.send_task(
            name,
            kwargs=payload,
            task_id=task_id,
        )
        return str(getattr(result, "id", task_id))


@dataclass
class CeleryBackedTaskQueue(LocalTaskQueue):
    publisher: CeleryTaskPublisher | None = None

    def __post_init__(self) -> None:
        if self.publisher is None:
            settings = get_settings()
            self.publisher = CeleryTaskPublisher(
                broker_url=settings.celery_broker_url,
                result_backend=settings.celery_result_backend,
            )

    def enqueue(
        self,
        name: str,
        payload: dict[str, Any],
        handler: Callable[[dict[str, Any]], T],
    ) -> TaskRecord[T]:
        task_id = str(uuid4())
        celery_task_id = self.publisher.publish_task(task_id, name, payload) if self.publisher else ""
        event_bus.publish_nowait(
            "task.enqueued",
            {
                "task_id": task_id,
                "name": name,
                "payload": payload,
                "celery_task_id": celery_task_id,
            },
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
                {"task_id": task_id, "name": name, "error": str(exc), "celery_task_id": celery_task_id},
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
            {"task_id": task_id, "name": name, "celery_task_id": celery_task_id},
        )
        return record

    def enqueue_deferred(self, name: str, payload: dict[str, Any]) -> TaskRecord[Any]:
        task_id = str(uuid4())
        celery_task_id = self.publisher.publish_task(task_id, name, payload) if self.publisher else ""
        record = TaskRecord(
            task_id=task_id,
            name=name,
            payload=payload,
            status="queued",
        )
        self._history.append(record)
        event_bus.publish_nowait(
            "task.enqueued",
            {
                "task_id": task_id,
                "name": name,
                "payload": payload,
                "celery_task_id": celery_task_id,
            },
        )
        return record


def get_task_queue() -> LocalTaskQueue:
    settings = get_settings()
    if settings.offline_task_backend == "celery":
        return CeleryBackedTaskQueue()
    if settings.offline_task_backend == "redis_stream":
        return RedisBackedTaskQueue()
    return LocalTaskQueue()


task_queue = get_task_queue()
