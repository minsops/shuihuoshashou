from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from typing import Any, Literal

import httpx

from libs.common.config import get_settings
from libs.schemas import TranscriptSegment

Speaker = Literal["interviewer", "candidate", "unknown"]


class ASREngine(ABC):
    @abstractmethod
    async def transcribe_chunk(
        self,
        session_id: str,
        seq: int,
        audio_b64: str,
        *,
        speaker: Speaker | None = None,
        start_ms: int | None = None,
        end_ms: int | None = None,
        is_final: bool = True,
        confidence: float | None = None,
    ) -> TranscriptSegment:
        raise NotImplementedError


class StubASREngine(ASREngine):
    async def transcribe_chunk(
        self,
        session_id: str,
        seq: int,
        audio_b64: str,
        *,
        speaker: Speaker | None = None,
        start_ms: int | None = None,
        end_ms: int | None = None,
        is_final: bool = True,
        confidence: float | None = None,
    ) -> TranscriptSegment:
        try:
            raw = base64.b64decode(audio_b64).decode("utf-8")
        except Exception:
            raw = ""
        resolved_start_ms = seq * 1000 if start_ms is None else start_ms
        return TranscriptSegment(
            session_id=session_id,
            speaker=speaker or ("candidate" if seq % 2 else "interviewer"),
            text=raw or "本地 ASR stub 收到音频片段，生产环境请替换为云 ASR 或 faster-whisper。",
            start_ms=resolved_start_ms,
            end_ms=resolved_start_ms + 900 if end_ms is None else end_ms,
            is_final=is_final,
            confidence=0.8 if confidence is None else confidence,
        )


class HTTPASREngine(ASREngine):
    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._transport = transport

    async def transcribe_chunk(
        self,
        session_id: str,
        seq: int,
        audio_b64: str,
        *,
        speaker: Speaker | None = None,
        start_ms: int | None = None,
        end_ms: int | None = None,
        is_final: bool = True,
        confidence: float | None = None,
    ) -> TranscriptSegment:
        settings = get_settings()
        if not settings.asr_base_url:
            raise RuntimeError("ASR_PROVIDER=http requires ASR_BASE_URL")
        url = settings.asr_base_url.rstrip("/") + "/" + settings.asr_api_path.lstrip("/")
        payload = {
            "session_id": session_id,
            "seq": seq,
            "audio": audio_b64,
            "speaker": speaker,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "is_final": is_final,
            "confidence": confidence,
        }
        headers = _asr_headers()
        async with httpx.AsyncClient(
            timeout=settings.asr_timeout_seconds,
            transport=self._transport,
        ) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
        data = response.json()
        resolved_start_ms = _extract_optional(data, settings.asr_start_ms_path, start_ms)
        resolved_end_ms = _extract_optional(data, settings.asr_end_ms_path, end_ms)
        if resolved_start_ms is None:
            resolved_start_ms = seq * 1000
        if resolved_end_ms is None:
            resolved_end_ms = resolved_start_ms + 900
        resolved_confidence = _extract_optional(data, settings.asr_confidence_path, confidence)
        if resolved_confidence is None:
            resolved_confidence = 0.0
        resolved_final = _extract_optional(data, settings.asr_is_final_path, is_final)
        return TranscriptSegment(
            session_id=session_id,
            speaker=_coerce_speaker(_extract_optional(data, settings.asr_speaker_path, speaker)),
            text=str(_extract_path(data, settings.asr_text_path)),
            start_ms=int(resolved_start_ms),
            end_ms=int(resolved_end_ms),
            is_final=bool(resolved_final),
            confidence=float(resolved_confidence),
        )


def _asr_headers() -> dict[str, str]:
    settings = get_settings()
    if not settings.asr_api_key:
        return {}
    auth_value = (
        f"{settings.asr_auth_scheme} {settings.asr_api_key}"
        if settings.asr_auth_scheme
        else settings.asr_api_key
    )
    return {settings.asr_auth_header: auth_value}


def _extract_path(payload: Any, path: str) -> Any:
    value = payload
    for part in path.split("."):
        if isinstance(value, list):
            value = value[int(part)]
        elif isinstance(value, dict):
            value = value[part]
        else:
            raise KeyError(path)
    return value


def _extract_optional(payload: Any, path: str, fallback: Any) -> Any:
    try:
        return _extract_path(payload, path)
    except (KeyError, IndexError, TypeError, ValueError):
        return fallback


def _coerce_speaker(value: Any) -> Speaker:
    if value in {"interviewer", "candidate", "unknown"}:
        return value
    return "unknown"


def get_asr_engine() -> ASREngine:
    settings = get_settings()
    if settings.asr_provider == "http":
        return HTTPASREngine()
    return StubASREngine()
