from __future__ import annotations

import asyncio
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from libs.common.config import Settings
from libs.schemas import TranscriptSegment
from scripts import create_aliyun_nls_token
from scripts import check_aliyun_nls_asr
from scripts.create_aliyun_nls_token import AliyunNLSToken
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

        assert isinstance(ws.sent[0], str)
        assert isinstance(ws.sent[1], bytes)
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


def test_check_aliyun_nls_asr_smoke_script_uses_pcm_file(
    tmp_path: Path, monkeypatch
) -> None:
    pcm_path = tmp_path / "sample.pcm"
    pcm_path.write_bytes(b"\x00" * 6400)
    monkeypatch.setenv("ALIYUN_NLS_APP_KEY", "nls-app-key")
    monkeypatch.setenv("ALIYUN_NLS_TOKEN", "nls-token")

    class FakeSmokeSession:
        instances: list["FakeSmokeSession"] = []

        def __init__(self, session_id: str, *, settings: Settings) -> None:
            self.session_id = session_id
            self.settings = settings
            self.result_queue: asyncio.Queue[TranscriptSegment | None] = asyncio.Queue()
            self.sent_audio: list[bytes] = []
            FakeSmokeSession.instances.append(self)

        async def connect(self) -> None:
            return None

        async def send_audio(self, pcm_bytes: bytes) -> None:
            self.sent_audio.append(pcm_bytes)

        async def close(self) -> None:
            await self.result_queue.put(
                TranscriptSegment(
                    session_id=self.session_id,
                    speaker="unknown",
                    text="水货杀手 NLS 语音识别测试",
                    start_ms=0,
                    end_ms=1000,
                    is_final=True,
                    confidence=0.92,
                )
            )

    monkeypatch.setattr(check_aliyun_nls_asr, "AliyunNLSSession", FakeSmokeSession)

    exit_code = asyncio.run(check_aliyun_nls_asr._run(pcm_path))

    assert exit_code == 0
    assert FakeSmokeSession.instances
    assert FakeSmokeSession.instances[0].sent_audio == [b"\x00" * 3200, b"\x00" * 3200]


def test_check_aliyun_nls_asr_smoke_script_requires_token(
    tmp_path: Path, monkeypatch
) -> None:
    pcm_path = tmp_path / "sample.pcm"
    pcm_path.write_bytes(b"\x00" * 3200)
    monkeypatch.setenv("ALIYUN_NLS_APP_KEY", "nls-app-key")
    monkeypatch.setenv("ALIYUN_NLS_TOKEN", "")

    exit_code = asyncio.run(check_aliyun_nls_asr._run(pcm_path))

    assert exit_code == 2


def test_check_aliyun_nls_asr_allow_empty_result_does_not_claim_transcript_verified(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    pcm_path = tmp_path / "sample.pcm"
    pcm_path.write_bytes(b"\x00" * 3200)
    monkeypatch.setenv("ALIYUN_NLS_APP_KEY", "nls-app-key")
    monkeypatch.setenv("ALIYUN_NLS_TOKEN", "nls-token")

    class FakeEmptySession:
        def __init__(self, session_id: str, *, settings: Settings) -> None:
            self.session_id = session_id
            self.settings = settings
            self.result_queue: asyncio.Queue[TranscriptSegment | None] = asyncio.Queue()

        async def connect(self) -> None:
            return None

        async def send_audio(self, pcm_bytes: bytes) -> None:
            return None

        async def close(self) -> None:
            return None

    monkeypatch.setattr(check_aliyun_nls_asr, "AliyunNLSSession", FakeEmptySession)

    exit_code = asyncio.run(check_aliyun_nls_asr._run(pcm_path, allow_empty_result=True))

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Aliyun NLS ASR session completed, but no transcript text was verified." in captured.out
    assert "Aliyun NLS ASR smoke test ok." not in captured.out


def test_check_aliyun_nls_asr_smoke_script_can_create_token_from_ak(
    tmp_path: Path, monkeypatch
) -> None:
    pcm_path = tmp_path / "sample.pcm"
    pcm_path.write_bytes(b"\x00" * 3200)
    monkeypatch.setenv("ALIYUN_NLS_APP_KEY", "nls-app-key")
    monkeypatch.setenv("ALIYUN_NLS_TOKEN", "")
    monkeypatch.setenv("ALIYUN_AK_ID", "ak-id")
    monkeypatch.setenv("ALIYUN_AK_SECRET", "ak-secret")
    monkeypatch.setattr(
        check_aliyun_nls_asr,
        "create_token_from_env",
        lambda: AliyunNLSToken(id="created-token", expire_time=1700000000),
    )

    class FakeSmokeSession:
        instances: list["FakeSmokeSession"] = []

        def __init__(self, session_id: str, *, settings: Settings) -> None:
            self.session_id = session_id
            self.settings = settings
            self.result_queue: asyncio.Queue[TranscriptSegment | None] = asyncio.Queue()
            FakeSmokeSession.instances.append(self)

        async def connect(self) -> None:
            return None

        async def send_audio(self, pcm_bytes: bytes) -> None:
            return None

        async def close(self) -> None:
            await self.result_queue.put(
                TranscriptSegment(
                    session_id=self.session_id,
                    speaker="unknown",
                    text="自动 Token 测试",
                    start_ms=0,
                    end_ms=1000,
                    is_final=True,
                    confidence=0.92,
                )
            )

    monkeypatch.setattr(check_aliyun_nls_asr, "AliyunNLSSession", FakeSmokeSession)

    exit_code = asyncio.run(check_aliyun_nls_asr._run(pcm_path))

    assert exit_code == 0
    assert FakeSmokeSession.instances[0].settings.aliyun_nls_token == "created-token"


def test_create_aliyun_nls_token_builds_signed_request(monkeypatch) -> None:
    calls: dict[str, str | float] = {}

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "Token": {
                        "Id": "nls-token-id",
                        "ExpireTime": 1700000000,
                        "UserId": "123",
                    }
                }
            ).encode("utf-8")

    def fake_urlopen(url: str, *, timeout: float, context) -> FakeResponse:
        del context
        calls["url"] = url
        calls["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(create_aliyun_nls_token, "urlopen", fake_urlopen)

    token = create_aliyun_nls_token.create_token(
        "ak-id",
        "ak-secret",
        endpoint="https://example.aliyun.com/",
        timeout=3,
    )

    assert token.id == "nls-token-id"
    assert token.expire_time == 1700000000
    assert calls["timeout"] == 3
    query = parse_qs(urlparse(str(calls["url"])).query)
    assert query["Action"] == ["CreateToken"]
    assert query["Version"] == ["2019-02-28"]
    assert query["RegionId"] == ["cn-shanghai"]
    assert query["SignatureMethod"] == ["HMAC-SHA1"]
    assert "Signature" in query
