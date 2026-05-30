from __future__ import annotations

import pytest

from libs.common.events import event_bus
from libs.common.tasks import LocalTaskQueue, RedisBackedTaskQueue, RedisStreamPublisher


class FakeRedis:
    def __init__(self) -> None:
        self.messages: list[tuple[str, dict[str, str]]] = []

    def xadd(self, stream: str, message: dict[str, str]) -> str:
        self.messages.append((stream, message))
        return "1700000000000-0"


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


def test_redis_stream_publisher_writes_task_message() -> None:
    client = FakeRedis()
    publisher = RedisStreamPublisher(
        redis_url="redis://localhost:6379/0",
        stream_prefix="test",
        client=client,
    )

    stream_id = publisher.publish_task("task-1", "interview.offline_scoring", {"interview_id": "abc"})

    assert stream_id == "1700000000000-0"
    assert client.messages
    stream, message = client.messages[0]
    assert stream == "test:tasks:interview.offline_scoring"
    assert message["task_id"] == "task-1"
    assert message["name"] == "interview.offline_scoring"
    assert '"interview_id": "abc"' in message["payload"]
    assert message["created_at"]


def test_redis_backed_task_queue_publishes_and_runs_handler() -> None:
    event_bus.reset()
    publisher = RedisStreamPublisher(
        redis_url="redis://localhost:6379/0",
        stream_prefix="test",
        client=FakeRedis(),
    )
    queue = RedisBackedTaskQueue(publisher=publisher)

    record = queue.enqueue("demo.task", {"value": 2}, lambda payload: payload["value"] + 3)

    assert record.status == "completed"
    assert record.result == 5
    assert queue.history("demo.task") == [record]
    history = event_bus.history()
    assert [topic for topic, _ in history] == ["task.enqueued", "task.completed"]
    assert history[0][1]["stream_id"] == "1700000000000-0"
