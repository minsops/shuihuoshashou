from __future__ import annotations

import pytest

from libs.common.events import event_bus
from libs.common.tasks import LocalTaskQueue


def test_local_task_queue_records_success() -> None:
    event_bus.reset()
    queue = LocalTaskQueue()

    record = queue.enqueue("demo.task", {"value": 2}, lambda payload: payload["value"] + 3)

    assert record.status == "completed"
    assert record.result == 5
    assert queue.history("demo.task") == [record]
    assert [topic for topic, _ in event_bus.history()] == ["task.enqueued", "task.completed"]


def test_local_task_queue_records_failure() -> None:
    event_bus.reset()
    queue = LocalTaskQueue()

    def fail(_: dict) -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        queue.enqueue("demo.task", {}, fail)

    records = queue.history("demo.task")
    assert records[0].status == "failed"
    assert records[0].error == "boom"
    assert [topic for topic, _ in event_bus.history()] == ["task.enqueued", "task.failed"]
