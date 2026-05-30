from __future__ import annotations

import base64
import asyncio

import httpx

from libs.common.config import get_settings
from libs.schemas import TranscriptSegment
from services.asr_service.service import (
    ASRSessionManager,
    HTTPASREngine,
    StubASREngine,
    get_asr_engine,
)


def test_stub_asr_decodes_text_and_metadata() -> None:
    engine = StubASREngine()
    audio = base64.b64encode("候选人的回答".encode("utf-8")).decode("ascii")

    segment = asyncio.run(engine.transcribe_chunk(
        "session-1",
        2,
        audio,
        speaker="candidate",
        start_ms=100,
        end_ms=900,
        is_final=False,
        confidence=0.7,
    ))

    assert segment.text == "候选人的回答"
    assert segment.speaker == "candidate"
    assert segment.start_ms == 100
    assert segment.end_ms == 900
    assert segment.is_final is False
    assert segment.confidence == 0.7


def test_http_asr_engine_posts_audio_and_maps_response(monkeypatch) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "result": {
                    "text": "我写了 FastAPI 的异常降级。",
                    "speaker": "candidate",
                    "start_ms": 120,
                    "end_ms": 980,
                    "is_final": True,
                    "confidence": 0.91,
                }
            },
        )

    monkeypatch.setenv("ASR_PROVIDER", "http")
    monkeypatch.setenv("ASR_BASE_URL", "https://asr.example.com/v1")
    monkeypatch.setenv("ASR_API_PATH", "/streaming/transcribe")
    monkeypatch.setenv("ASR_API_KEY", "asr-secret")
    monkeypatch.setenv("ASR_TEXT_PATH", "result.text")
    monkeypatch.setenv("ASR_SPEAKER_PATH", "result.speaker")
    monkeypatch.setenv("ASR_START_MS_PATH", "result.start_ms")
    monkeypatch.setenv("ASR_END_MS_PATH", "result.end_ms")
    monkeypatch.setenv("ASR_IS_FINAL_PATH", "result.is_final")
    monkeypatch.setenv("ASR_CONFIDENCE_PATH", "result.confidence")
    get_settings.cache_clear()
    engine = HTTPASREngine(transport=httpx.MockTransport(handler))

    segment = asyncio.run(engine.transcribe_chunk(
        "session-1",
        7,
        "YXVkaW8=",
        speaker="candidate",
        start_ms=100,
        end_ms=900,
        is_final=False,
        confidence=0.5,
    ))

    assert segment.text == "我写了 FastAPI 的异常降级。"
    assert segment.speaker == "candidate"
    assert segment.start_ms == 120
    assert segment.end_ms == 980
    assert segment.is_final is True
    assert segment.confidence == 0.91
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "https://asr.example.com/v1/streaming/transcribe"
    assert request.headers["Authorization"] == "Bearer asr-secret"
    assert b'"seq":7' in request.content
    assert b'"audio":"YXVkaW8="' in request.content


def test_get_asr_engine_uses_http_provider(monkeypatch) -> None:
    monkeypatch.setenv("ASR_PROVIDER", "http")
    monkeypatch.setenv("ASR_BASE_URL", "https://asr.example.com")
    get_settings.cache_clear()

    assert isinstance(get_asr_engine(), HTTPASREngine)


def test_asr_session_manager_accepts_partial_then_final_same_seq() -> None:
    manager = ASRSessionManager()
    partial = TranscriptSegment(
        session_id="session-1",
        speaker="candidate",
        text="我写了",
        start_ms=0,
        end_ms=400,
        is_final=False,
        confidence=0.6,
    )
    final = partial.model_copy(update={"text": "我写了 FastAPI 编排。", "is_final": True})

    first = manager.accept_segment(1, partial)
    second = manager.accept_segment(1, final)
    duplicate = manager.accept_segment(1, final)

    assert first.accepted is True
    assert second.accepted is True
    assert second.segment is not None
    assert second.segment.text == "我写了 FastAPI 编排。"
    assert duplicate.accepted is False
    assert duplicate.reason == "duplicate_final_segment"


def test_asr_session_manager_smooths_unknown_speaker_by_continuity() -> None:
    manager = ASRSessionManager(speaker_continuity_gap_ms=1000)
    known = TranscriptSegment(
        session_id="session-1",
        speaker="candidate",
        text="第一段",
        start_ms=0,
        end_ms=500,
        is_final=True,
        confidence=0.9,
    )
    unknown = TranscriptSegment(
        session_id="session-1",
        speaker="unknown",
        text="连续补充",
        start_ms=800,
        end_ms=1200,
        is_final=True,
        confidence=0.7,
    )

    manager.accept_segment(1, known)
    decision = manager.accept_segment(2, unknown)

    assert decision.accepted is True
    assert decision.segment is not None
    assert decision.segment.speaker == "candidate"


def test_asr_session_manager_resolves_unknown_speaker_from_audio_cluster() -> None:
    manager = ASRSessionManager(speaker_continuity_gap_ms=0)
    audio = base64.b64encode(b"candidate-voiceprint").decode("ascii")
    known = TranscriptSegment(
        session_id="session-1",
        speaker="candidate",
        text="第一段",
        start_ms=0,
        end_ms=500,
        is_final=True,
        confidence=0.9,
    )
    unknown_same_voice = TranscriptSegment(
        session_id="session-1",
        speaker="unknown",
        text="同一个说话人稍后补充",
        start_ms=5000,
        end_ms=5600,
        is_final=True,
        confidence=0.7,
    )

    manager.accept_segment(1, known, audio_b64=audio)
    decision = manager.accept_segment(2, unknown_same_voice, audio_b64=audio)

    assert decision.accepted is True
    assert decision.segment is not None
    assert decision.segment.speaker == "candidate"
