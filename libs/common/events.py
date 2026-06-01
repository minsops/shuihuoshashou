from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

from libs.common.observability import metrics_registry


class LocalEventBus:
    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[dict[str, Any]]] = defaultdict(asyncio.Queue)
        self._history: list[tuple[str, dict[str, Any]]] = []

    async def publish(self, topic: str, event: dict[str, Any]) -> None:
        self.publish_nowait(topic, event)

    def publish_nowait(self, topic: str, event: dict[str, Any]) -> None:
        self._history.append((topic, event))
        metrics_registry.record_event(topic)
        self._queues[topic].put_nowait(event)

    async def next_event(self, topic: str) -> dict[str, Any]:
        return await self._queues[topic].get()

    def history(self, topic: str | None = None) -> list[tuple[str, dict[str, Any]]]:
        if topic is None:
            return list(self._history)
        return [(item_topic, event) for item_topic, event in self._history if item_topic == topic]

    def reset(self) -> None:
        self._queues.clear()
        self._history.clear()


event_bus = LocalEventBus()
