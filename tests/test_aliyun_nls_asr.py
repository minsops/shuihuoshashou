from __future__ import annotations

import asyncio
import json

from libs.common.config import Settings
from services.asr_service.nls_engine import AliyunNLSSession


class FakeNLSWS:
    def __init__(self, *, fail_on_start: bool = False, fail_on_audio: bool = False) -> None:
        self.fail_on_start = fail_on_start
        self.fail_on_audio = fail_on_audio
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
                                "name": "TaskFailed",
                                "status": 40000001,
                                "status_text": "bad audio",
                            }
                        }
                    )
                )
                return
            await self._events.put(
                json.dumps(
                    {
                        "header": {"name": "SentenceEnd", "status": 20000000},
                        "payload": {
                            "result": "好，我知道了",
                            "begin_time": 170,
                            "time": 920,
                            "confidence": 0.91,
                        },
                    }
                )
            )
            return
        data = json.loads(message)
        name = data.get("header", {}).get("name")
        if name == "StartTranscription":
            if self.fail_on_start:
                await self._events.put(
                    json.dumps(
                        {
                            "header": {
                                "name": "TaskFailed",
                                "status": 40000002,
                                "status_text": "bad token",
                            }
                        }
                    )
                )
                return
            await self._events.put(
                json.dumps({"header": {"name": "TranscriptionStarted", "status": 20000000}})
            )
        elif name == "StopTranscription":
            await self._events.put(
                json.dumps({"header": {"name": "TranscriptionCompleted", "status": 20000000}})
            )

    async def recv(self) -> str:
        return await self._events.get()

    async def close(self) -> None:
        self.closed = True


def _settings() -> Settings:
    return Settings(
        asr_provider="aliyun_nls_ws",
        aliyun_nls_app_key="nls-app-key",
        aliyun_nls_token="nls-token",
    )


def test_aliyun_nls_session_lifecycle() -> None:
    async def run() -> None:
        ws = FakeNLSWS()
        session = AliyunNLSSession("interview-1", settings=_settings(), ws=ws)

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
        assert segment.confidence == 0.91
        assert ws.closed is True

    asyncio.run(run())


def test_aliyun_nls_session_closes_websocket_when_start_fails() -> None:
    async def run() -> None:
        ws = FakeNLSWS(fail_on_start=True)
        session = AliyunNLSSession("interview-1", settings=_settings(), ws=ws)

        try:
            await session.connect()
        except RuntimeError as exc:
            assert "aliyun_asr_task_failed:40000002:bad token" in str(exc)
        else:
            raise AssertionError("connect should fail")

        assert session.error_reason == "aliyun_asr_task_failed:40000002:bad token"
        assert ws.closed is True
        assert session.ws is None

    asyncio.run(run())
