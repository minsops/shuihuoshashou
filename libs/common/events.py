from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any


class LocalEventBus:
    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[dict[str, Any]]] = defaultdict(asyncio.Queue)

    async def publish(self, topic: str, event: dict[str, Any]) -> None:
        await self._queues[topic].put(event)

    async def next_event(self, topic: str) -> dict[str, Any]:
        return await self._queues[topic].get()


event_bus = LocalEventBus()

