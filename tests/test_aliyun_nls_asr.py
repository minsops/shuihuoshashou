from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from libs.common.config import Settings
from libs.schemas import TranscriptSegment
from scripts import create_aliyun_nls_token
from scripts import check_aliyun_nls_asr
from scripts.create_aliyun_nls_token import AliyunNLSToken
from services.asr_service import nls_token
from services.asr_service.nls_engine import AliyunNLSSession, AliyunNLSWSASREngine


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

    class FakeSmokeEngine:
        def __init__(self, *, settings: Settings) -> None:
            self.settings = settings

        async def get_or_create_session(self, session_id: str) -> FakeSmokeSession:
            return FakeSmokeSession(session_id, settings=self.settings)

        async def close_session(self, session_id: str) -> None:
            return None

    monkeypatch.setattr(check_aliyun_nls_asr, "AliyunNLSWSASREngine", FakeSmokeEngine)

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


def test_check_aliyun_nls_asr_help_clarifies_allow_empty_result() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_aliyun_nls_asr.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    help_text = " ".join(result.stdout.split())
    assert "Exit 0 when the ASR session completes but no transcript text is returned." in help_text
    assert "Treat a completed ASR session with no transcript text as success." not in help_text


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

    class FakeEmptyEngine:
        def __init__(self, *, settings: Settings) -> None:
            self.settings = settings

        async def get_or_create_session(self, session_id: str) -> FakeEmptySession:
            return FakeEmptySession(session_id, settings=self.settings)

        async def close_session(self, session_id: str) -> None:
            return None

    monkeypatch.setattr(check_aliyun_nls_asr, "AliyunNLSWSASREngine", FakeEmptyEngine)

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

    class FakeSmokeEngine:
        def __init__(self, *, settings: Settings) -> None:
            self.settings = settings

        async def get_or_create_session(self, session_id: str) -> FakeSmokeSession:
            return FakeSmokeSession(session_id, settings=self.settings)

        async def close_session(self, session_id: str) -> None:
            return None

    monkeypatch.setattr(check_aliyun_nls_asr, "AliyunNLSWSASREngine", FakeSmokeEngine)

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

    monkeypatch.setattr(nls_token, "urlopen", fake_urlopen)

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


def test_nls_token_provider_reuses_fresh_token() -> None:
    created: list[str] = []

    def fake_create_token(
        access_key_id: str,
        access_key_secret: str,
        *,
        endpoint: str,
        region_id: str,
    ) -> AliyunNLSToken:
        assert access_key_id == "ak-id"
        assert access_key_secret == "ak-secret"
        assert endpoint == "https://example.aliyun.com/"
        assert region_id == "cn-test"
        created.append(f"created-token-{len(created) + 1}")
        return AliyunNLSToken(id=created[-1], expire_time=2000)

    provider = nls_token.AliyunNLSTokenProvider(
        access_key_id="ak-id",
        access_key_secret="ak-secret",
        endpoint="https://example.aliyun.com/",
        region_id="cn-test",
        create_token_func=fake_create_token,
        now_func=lambda: 1000,
    )

    assert provider.get_token() == "created-token-1"
    assert provider.get_token() == "created-token-1"
    assert created == ["created-token-1"]


def test_nls_token_provider_refreshes_expiring_token() -> None:
    created: list[str] = []

    def fake_create_token(*args, **kwargs) -> AliyunNLSToken:
        del args, kwargs
        created.append(f"created-token-{len(created) + 1}")
        return AliyunNLSToken(id=created[-1], expire_time=1100)

    provider = nls_token.AliyunNLSTokenProvider(
        access_key_id="ak-id",
        access_key_secret="ak-secret",
        create_token_func=fake_create_token,
        now_func=lambda: 1000,
    )

    assert provider.get_token() == "created-token-1"
    assert provider.get_token() == "created-token-2"
    assert created == ["created-token-1", "created-token-2"]


def test_nls_token_provider_invalidate_forces_refresh() -> None:
    created: list[str] = []

    def fake_create_token(*args, **kwargs) -> AliyunNLSToken:
        del args, kwargs
        created.append(f"created-token-{len(created) + 1}")
        return AliyunNLSToken(id=created[-1], expire_time=2000)

    provider = nls_token.AliyunNLSTokenProvider(
        access_key_id="ak-id",
        access_key_secret="ak-secret",
        create_token_func=fake_create_token,
        now_func=lambda: 1000,
    )

    assert provider.get_token() == "created-token-1"
    provider.invalidate()
    assert provider.get_token() == "created-token-2"
    assert created == ["created-token-1", "created-token-2"]


def test_nls_session_gets_auto_token_from_provider() -> None:
    class FakeTokenProvider:
        def __init__(self) -> None:
            self.calls = 0

        def get_token(self) -> str:
            self.calls += 1
            return "created-token"

    provider = FakeTokenProvider()
    session = AliyunNLSSession(
        "session-1",
        settings=Settings(
            asr_provider="aliyun_nls_ws",
            aliyun_nls_app_key="app-key",
            aliyun_nls_token="",
            aliyun_ak_id="ak-id",
            aliyun_ak_secret="ak-secret",
            aliyun_nls_token_endpoint="https://example.aliyun.com/",
            aliyun_nls_token_region="cn-test",
        ),
        token_provider=provider,
    )

    assert asyncio.run(session._resolve_token()) == "created-token"
    assert provider.calls == 1


def test_nls_engine_shares_auto_token_provider_across_sessions(monkeypatch) -> None:
    async def fake_connect(self) -> None:
        self.started = True

    monkeypatch.setattr(AliyunNLSSession, "connect", fake_connect)
    engine = AliyunNLSWSASREngine(
        settings=Settings(
            asr_provider="aliyun_nls_ws",
            aliyun_nls_app_key="app-key",
            aliyun_nls_token="",
            aliyun_ak_id="ak-id",
            aliyun_ak_secret="ak-secret",
        )
    )

    first = asyncio.run(engine.get_or_create_session("session-1"))
    second = asyncio.run(engine.get_or_create_session("session-2"))

    assert engine._token_provider is not None
    assert first._token_provider is engine._token_provider
    assert second._token_provider is engine._token_provider


def test_nls_engine_retries_once_after_auto_token_failure(monkeypatch) -> None:
    connect_calls = 0

    class FakeTokenProvider:
        def __init__(self) -> None:
            self.invalidations = 0

        def get_token(self) -> str:
            return "created-token"

        def invalidate(self) -> None:
            self.invalidations += 1

    async def fake_connect(self) -> None:
        nonlocal connect_calls
        connect_calls += 1
        if connect_calls == 1:
            raise RuntimeError("aliyun_asr_task_failed:40000002:bad token")
        self.started = True

    monkeypatch.setattr(AliyunNLSSession, "connect", fake_connect)
    provider = FakeTokenProvider()
    engine = AliyunNLSWSASREngine(
        settings=Settings(
            asr_provider="aliyun_nls_ws",
            aliyun_nls_app_key="app-key",
            aliyun_nls_token="",
            aliyun_ak_id="ak-id",
            aliyun_ak_secret="ak-secret",
        )
    )
    engine._token_provider = provider

    session = asyncio.run(engine.get_or_create_session("session-1"))

    assert session.started is True
    assert provider.invalidations == 1
    assert connect_calls == 2
    assert engine._sessions["session-1"] is session


def test_nls_engine_retries_with_auto_token_after_fixed_token_403(monkeypatch) -> None:
    connect_tokens: list[str] = []
    connect_auto_flags: list[bool] = []

    class FakeTokenProvider:
        def __init__(self) -> None:
            self.invalidations = 0

        def get_token(self) -> str:
            return "created-token"

        def invalidate(self) -> None:
            self.invalidations += 1

    async def fake_connect(self) -> None:
        connect_auto_flags.append(self._prefer_auto_token)
        connect_tokens.append(await self._resolve_token())
        if len(connect_tokens) == 1:
            raise RuntimeError("server rejected WebSocket connection: HTTP 403")
        self.started = True

    monkeypatch.setattr(AliyunNLSSession, "connect", fake_connect)
    provider = FakeTokenProvider()
    engine = AliyunNLSWSASREngine(
        settings=Settings(
            asr_provider="aliyun_nls_ws",
            aliyun_nls_app_key="app-key",
            aliyun_nls_token="expired-fixed-token",
            aliyun_ak_id="ak-id",
            aliyun_ak_secret="ak-secret",
        )
    )
    engine._token_provider = provider

    session = asyncio.run(engine.get_or_create_session("session-1"))

    assert session.started is True
    assert provider.invalidations == 1
    assert connect_auto_flags == [False, True]
    assert connect_tokens == ["expired-fixed-token", "created-token"]
    assert engine._prefer_auto_token is True
    assert engine._sessions["session-1"] is session
