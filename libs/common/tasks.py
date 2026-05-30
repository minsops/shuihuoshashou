from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar
from uuid import uuid4

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


task_queue = LocalTaskQueue()
