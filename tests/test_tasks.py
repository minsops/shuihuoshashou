from __future__ import annotations

import pytest

from libs.common.events import event_bus
from libs.common.observability import metrics_registry
from libs.common.tasks import (
    CeleryBackedTaskQueue,
    CeleryTaskPublisher,
    LocalTaskQueue,
    RedisBackedTaskQueue,
    RedisStreamPublisher,
    RedisStreamWorker,
)


class FakeRedis:
    def __init__(self) -> None:
        self.messages: list[tuple[str, dict[str, str]]] = []
        self.groups: list[tuple[str, str, str, bool]] = []
        self.acked: list[tuple[str, str, str]] = []

    def xadd(self, stream: str, message: dict[str, str]) -> str:
        self.messages.append((stream, message))
        return "1700000000000-0"

    def xgroup_create(self, stream: str, group: str, id: str, mkstream: bool) -> None:
        self.groups.append((stream, group, id, mkstream))

    def xreadgroup(
        self,
        group: str,
        consumer: str,
        streams: dict[str, str],
        count: int,
        block: int,
    ) -> list[tuple[str, list[tuple[str, dict[str, str]]]]]:
        if not self.messages:
            return []
        stream, message = self.messages.pop(0)
        assert streams == {stream: ">"}
        assert group
        assert consumer
        assert count == 1
        assert block == 1000
        return [(stream, [("1700000000000-0", message)])]

    def xack(self, stream: str, group: str, message_id: str) -> None:
        self.acked.append((stream, group, message_id))


class FakeCeleryResult:
    id = "celery-task-1"


class FakeCelerySender:
    def __init__(self) -> None:
        self.sent: list[tuple[str, dict]] = []

    def send_task(
        self,
        name: str,
        kwargs: dict,
        task_id: str,
        queue: str,
    ) -> FakeCeleryResult:
        self.sent.append((name, {"kwargs": kwargs, "task_id": task_id, "queue": queue}))
        return FakeCeleryResult()


class FakeCeleryApp:
    def __init__(self) -> None:
        self.tasks: dict[str, object] = {}

    def task(self, name: str):
        def decorate(func):
            self.tasks[name] = func
            return func

        return decorate


def test_local_task_queue_records_success() -> None:
    event_bus.reset()
    metrics_registry.reset()
    queue = LocalTaskQueue()

    record = queue.enqueue("demo.task", {"value": 2}, lambda payload: payload["value"] + 3)

    assert record.status == "completed"
    assert record.result == 5
    assert queue.history("demo.task") == [record]
    assert [topic for topic, _ in event_bus.history()] == ["task.enqueued", "task.completed"]
    metrics = metrics_registry.render_prometheus()
    assert 'shuihuo_events_total{topic="task.enqueued"} 1' in metrics
    assert 'shuihuo_events_total{topic="task.completed"} 1' in metrics


def test_local_task_queue_records_failure() -> None:
    event_bus.reset()
    metrics_registry.reset()
    queue = LocalTaskQueue()

    def fail(_: dict) -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        queue.enqueue("demo.task", {}, fail)

    records = queue.history("demo.task")
    assert records[0].status == "failed"
    assert records[0].error == "boom"
    assert [topic for topic, _ in event_bus.history()] == ["task.enqueued", "task.failed"]
    metrics = metrics_registry.render_prometheus()
    assert 'shuihuo_events_total{topic="task.enqueued"} 1' in metrics
    assert 'shuihuo_events_total{topic="task.failed"} 1' in metrics


def test_local_task_queue_can_defer_without_handler() -> None:
    event_bus.reset()
    queue = LocalTaskQueue()

    record = queue.enqueue_deferred("demo.task", {"value": 2})

    assert record.status == "queued"
    assert record.result is None
    assert queue.history("demo.task") == [record]
    assert [topic for topic, _ in event_bus.history()] == ["task.enqueued"]


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


def test_redis_backed_task_queue_can_defer_to_stream() -> None:
    event_bus.reset()
    publisher = RedisStreamPublisher(
        redis_url="redis://localhost:6379/0",
        stream_prefix="test",
        client=FakeRedis(),
    )
    queue = RedisBackedTaskQueue(publisher=publisher)

    record = queue.enqueue_deferred("demo.task", {"value": 2})

    assert record.status == "queued"
    assert record.result is None
    assert queue.history("demo.task") == [record]
    history = event_bus.history()
    assert [topic for topic, _ in history] == ["task.enqueued"]
    assert history[0][1]["stream_id"] == "1700000000000-0"


def test_redis_stream_worker_consumes_and_acks_task() -> None:
    event_bus.reset()
    client = FakeRedis()
    publisher = RedisStreamPublisher(
        redis_url="redis://localhost:6379/0",
        stream_prefix="test",
        client=client,
    )
    publisher.publish_task("task-1", "interview.offline_scoring", {"interview_id": "abc"})
    handled: list[dict] = []
    worker = RedisStreamWorker(
        "interview.offline_scoring",
        handled.append,
        redis_url="redis://localhost:6379/0",
        stream_prefix="test",
        client=client,
    )

    processed = worker.consume_once()

    assert processed == 1
    assert handled == [{"interview_id": "abc"}]
    assert client.groups == [
        ("test:tasks:interview.offline_scoring", "offline-workers", "0", True)
    ]
    assert client.acked == [
        ("test:tasks:interview.offline_scoring", "offline-workers", "1700000000000-0")
    ]
    assert [topic for topic, _ in event_bus.history()] == ["task.worker_completed"]


def test_redis_stream_worker_publishes_failure_without_ack() -> None:
    event_bus.reset()
    metrics_registry.reset()
    client = FakeRedis()
    publisher = RedisStreamPublisher(
        redis_url="redis://localhost:6379/0",
        stream_prefix="test",
        client=client,
    )
    publisher.publish_task("task-1", "interview.offline_scoring", {"interview_id": "abc"})

    def fail(_: dict) -> None:
        raise RuntimeError("boom")

    worker = RedisStreamWorker(
        "interview.offline_scoring",
        fail,
        redis_url="redis://localhost:6379/0",
        stream_prefix="test",
        client=client,
    )

    with pytest.raises(RuntimeError, match="boom"):
        worker.consume_once()

    assert client.acked == []
    history = event_bus.history()
    assert [topic for topic, _ in history] == ["task.worker_failed"]
    assert history[0][1]["error"] == "boom"
    metrics = metrics_registry.render_prometheus()
    assert 'shuihuo_events_total{topic="task.worker_failed"} 1' in metrics


def test_redis_stream_worker_reports_malformed_payload_without_ack() -> None:
    event_bus.reset()
    metrics_registry.reset()
    client = FakeRedis()
    client.messages.append(
        (
            "test:tasks:interview.offline_scoring",
            {
                "task_id": "task-1",
                "name": "interview.offline_scoring",
                "payload": "{not-json",
            },
        )
    )
    handled: list[dict] = []
    worker = RedisStreamWorker(
        "interview.offline_scoring",
        handled.append,
        redis_url="redis://localhost:6379/0",
        stream_prefix="test",
        client=client,
    )

    with pytest.raises(ValueError):
        worker.consume_once()

    assert handled == []
    assert client.acked == []
    history = event_bus.history()
    assert [topic for topic, _ in history] == ["task.worker_failed"]
    assert history[0][1]["task_id"] == "task-1"
    assert history[0][1]["name"] == "interview.offline_scoring"
    metrics = metrics_registry.render_prometheus()
    assert 'shuihuo_events_total{topic="task.worker_failed"} 1' in metrics


def test_celery_task_publisher_sends_named_task() -> None:
    sender = FakeCelerySender()
    publisher = CeleryTaskPublisher(
        broker_url="redis://localhost:6379/1",
        result_backend="redis://localhost:6379/2",
        sender=sender,
    )

    celery_task_id = publisher.publish_task(
        "task-1",
        "interview.offline_scoring",
        {"interview_id": "abc"},
    )

    assert celery_task_id == "celery-task-1"
    assert sender.sent == [
        (
            "interview.offline_scoring",
            {
                "kwargs": {"interview_id": "abc"},
                "task_id": "task-1",
                "queue": "shuihuo-offline",
            },
        )
    ]


def test_celery_task_publisher_uses_configured_queue() -> None:
    sender = FakeCelerySender()
    publisher = CeleryTaskPublisher(
        broker_url="redis://localhost:6379/1",
        result_backend="redis://localhost:6379/2",
        queue_name="custom-offline",
        sender=sender,
    )

    publisher.publish_task("task-1", "interview.offline_scoring", {"interview_id": "abc"})

    assert sender.sent[0][1]["queue"] == "custom-offline"


def test_celery_backed_task_queue_defers_to_celery() -> None:
    event_bus.reset()
    sender = FakeCelerySender()
    publisher = CeleryTaskPublisher(
        broker_url="redis://localhost:6379/1",
        result_backend="redis://localhost:6379/2",
        sender=sender,
    )
    queue = CeleryBackedTaskQueue(publisher=publisher)

    record = queue.enqueue_deferred("interview.offline_scoring", {"interview_id": "abc"})

    assert record.status == "queued"
    assert sender.sent[0][0] == "interview.offline_scoring"
    history = event_bus.history()
    assert [topic for topic, _ in history] == ["task.enqueued"]
    assert history[0][1]["celery_task_id"] == "celery-task-1"


def test_celery_backed_task_queue_publishes_and_runs_handler() -> None:
    event_bus.reset()
    sender = FakeCelerySender()
    publisher = CeleryTaskPublisher(
        broker_url="redis://localhost:6379/1",
        result_backend="redis://localhost:6379/2",
        sender=sender,
    )
    queue = CeleryBackedTaskQueue(publisher=publisher)

    record = queue.enqueue(
        "interview.offline_scoring",
        {"value": 2},
        lambda payload: payload["value"] + 3,
    )

    assert record.status == "completed"
    assert record.result == 5
    assert sender.sent[0][0] == "interview.offline_scoring"
    history = event_bus.history()
    assert [topic for topic, _ in history] == ["task.enqueued", "task.completed"]
    assert history[0][1]["celery_task_id"] == "celery-task-1"
    assert history[1][1]["celery_task_id"] == "celery-task-1"


def test_celery_task_registration_uses_offline_scoring_task_name() -> None:
    from services.offline_worker.celery_tasks import OFFLINE_SCORING_TASK, register_tasks

    app = FakeCeleryApp()
    task_func = register_tasks(app)

    assert app.tasks[OFFLINE_SCORING_TASK] is task_func
