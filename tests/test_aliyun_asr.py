from __future__ import annotations

import asyncio
import json

from libs.common.config import Settings
from services.asr_service.aliyun_engine import AliyunASRSession


class FakeAliyunWS:
    def __init__(self, *, fail_on_audio: bool = False, heartbeat_first: bool = False) -> None:
        self.fail_on_audio = fail_on_audio
        self.heartbeat_first = heartbeat_first
        self.closed = False
        self.sent: list[str | bytes] = []
        self._events: asyncio.Queue[str] = asyncio.Queue()

    async def send(self, message: str | bytes) -> None:
        self.sent.append(message)
        if isinstance(message, bytes):
            if self.fail_on_audio:
                await self._events.put(
                    json.dumps(
                        {
                            "header": {
                                "event": "task-failed",
                                "error_code": "InvalidAudio",
                                "error_message": "bad audio",
                            }
                        }
                    )
                )
                return
            if self.heartbeat_first:
                await self._events.put(
                    json.dumps(
                        {
                            "header": {"event": "result-generated"},
                            "payload": {
                                "output": {"sentence": {"heartbeat": True, "text": ""}}
                            },
                        }
                    )
                )
            await self._events.put(
                json.dumps(
                    {
                        "header": {"event": "result-generated"},
                        "payload": {
                            "output": {
                                "sentence": {
                                    "begin_time": 170,
                                    "end_time": 920,
                                    "text": "好，我知道了",
                                    "sentence_end": True,
                                }
                            }
                        },
                    }
                )
            )
            return
        data = json.loads(message)
        action = data.get("header", {}).get("action")
        if action == "run-task":
            await self._events.put(json.dumps({"header": {"event": "task-started"}}))
        elif action == "finish-task":
            await self._events.put(json.dumps({"header": {"event": "task-finished"}}))

    async def recv(self) -> str:
        return await self._events.get()

    async def close(self) -> None:
        self.closed = True


def _settings() -> Settings:
    return Settings(asr_provider="aliyun_ws", aliyun_asr_api_key="dashscope-secret")


def test_aliyun_session_lifecycle() -> None:
    async def run() -> None:
        ws = FakeAliyunWS()
        session = AliyunASRSession("interview-1", settings=_settings(), ws=ws)

        await session.connect()
        await session.send_audio(b"\x00" * 3200)
        segment = await asyncio.wait_for(session.result_queue.get(), timeout=2)
        await session.close()

        assert segment is not None
        assert segment.session_id == "interview-1"
        assert segment.speaker == "unknown"
        assert segment.text == "好，我知道了"
        assert segment.start_ms == 170
        assert segment.end_ms == 920
        assert segment.is_final is True
        assert ws.closed is True

    asyncio.run(run())


def test_aliyun_session_skips_heartbeat() -> None:
    async def run() -> None:
        session = AliyunASRSession(
            "interview-1",
            settings=_settings(),
            ws=FakeAliyunWS(heartbeat_first=True),
        )

        await session.connect()
        await session.send_audio(b"\x00" * 3200)
        segment = await asyncio.wait_for(session.result_queue.get(), timeout=2)
        await session.close()

        assert segment is not None
        assert segment.text == "好，我知道了"

    asyncio.run(run())


def test_aliyun_session_handles_task_failed() -> None:
    async def run() -> None:
        session = AliyunASRSession(
            "interview-1",
            settings=_settings(),
            ws=FakeAliyunWS(fail_on_audio=True),
        )

        await session.connect()
        await session.send_audio(b"\x00" * 3200)
        item = await asyncio.wait_for(session.result_queue.get(), timeout=2)
        await session.close()

        assert item is None
        assert session.error_reason == "aliyun_asr_task_failed:InvalidAudio:bad audio"

    asyncio.run(run())
