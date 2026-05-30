from __future__ import annotations

import base64
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Literal

import httpx

from libs.common.config import get_settings
from libs.schemas import TranscriptSegment

Speaker = Literal["interviewer", "candidate", "unknown"]


@dataclass
class ASRSessionChunk:
    seq: int
    segment: TranscriptSegment


@dataclass
class ASRSessionDecision:
    accepted: bool
    segment: TranscriptSegment | None = None
    reason: str = ""


@dataclass
class SpeakerDiarizationState:
    cluster_roles: dict[str, Speaker] = field(default_factory=dict)


class SpeakerDiarizer(ABC):
    @abstractmethod
    def resolve_speaker(
        self,
        session_id: str,
        audio_b64: str | None,
        speaker: Speaker,
    ) -> Speaker:
        raise NotImplementedError

    def close(self, session_id: str) -> None:
        return None

    def reset(self) -> None:
        return None


class LocalSpeakerDiarizer(SpeakerDiarizer):
    def __init__(self) -> None:
        self._sessions: dict[str, SpeakerDiarizationState] = {}

    def resolve_speaker(
        self,
        session_id: str,
        audio_b64: str | None,
        speaker: Speaker,
    ) -> Speaker:
        fingerprint = _audio_fingerprint(audio_b64)
        if fingerprint is None:
            return speaker

        state = self._sessions.setdefault(session_id, SpeakerDiarizationState())
        if speaker in {"interviewer", "candidate"}:
            state.cluster_roles[fingerprint] = speaker
            return speaker
        return state.cluster_roles.get(fingerprint, speaker)

    def close(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def reset(self) -> None:
        self._sessions.clear()


@dataclass
class ASRSessionState:
    chunks: dict[int, ASRSessionChunk] = field(default_factory=dict)
    last_final_seq: int = -1
    last_known_speaker: Speaker | None = None
    last_known_end_ms: int | None = None


class ASRSessionManager:
    def __init__(
        self,
        speaker_continuity_gap_ms: int = 1500,
        speaker_diarizer: SpeakerDiarizer | None = None,
    ) -> None:
        self.speaker_continuity_gap_ms = speaker_continuity_gap_ms
        self.speaker_diarizer = speaker_diarizer or LocalSpeakerDiarizer()
        self._sessions: dict[str, ASRSessionState] = {}
        self._lock = Lock()

    def accept_segment(
        self,
        seq: int,
        segment: TranscriptSegment,
        *,
        audio_b64: str | None = None,
    ) -> ASRSessionDecision:
        with self._lock:
            state = self._sessions.setdefault(segment.session_id, ASRSessionState())
            existing = state.chunks.get(seq)
            if existing is not None and existing.segment.is_final and segment.is_final:
                return ASRSessionDecision(False, reason="duplicate_final_segment")
            if segment.is_final and seq <= state.last_final_seq:
                return ASRSessionDecision(False, reason="late_final_segment")

            resolved_speaker = self.speaker_diarizer.resolve_speaker(
                segment.session_id,
                audio_b64,
                segment.speaker,
            )
            if resolved_speaker != segment.speaker:
                segment = segment.model_copy(update={"speaker": resolved_speaker})
            segment = self._smooth_speaker(state, segment)
            state.chunks[seq] = ASRSessionChunk(seq=seq, segment=segment)
            if segment.is_final:
                state.last_final_seq = max(state.last_final_seq, seq)
            if segment.speaker in {"interviewer", "candidate"}:
                state.last_known_speaker = segment.speaker
                state.last_known_end_ms = segment.end_ms
            return ASRSessionDecision(True, segment=segment)

    def close(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
            self.speaker_diarizer.close(session_id)

    def reset(self) -> None:
        with self._lock:
            self._sessions.clear()
            self.speaker_diarizer.reset()

    def _smooth_speaker(
        self,
        state: ASRSessionState,
        segment: TranscriptSegment,
    ) -> TranscriptSegment:
        if segment.speaker != "unknown" or state.last_known_speaker is None:
            return segment
        if state.last_known_end_ms is None:
            return segment
        gap_ms = segment.start_ms - state.last_known_end_ms
        if 0 <= gap_ms <= self.speaker_continuity_gap_ms:
            return segment.model_copy(update={"speaker": state.last_known_speaker})
        return segment


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


def _audio_fingerprint(audio_b64: str | None) -> str | None:
    if not audio_b64:
        return None
    try:
        raw = base64.b64decode(audio_b64)
    except Exception:
        return None
    if not raw:
        return None
    return hashlib.sha256(raw).hexdigest()


def get_asr_engine() -> ASREngine:
    settings = get_settings()
    if settings.asr_provider == "http":
        return HTTPASREngine()
    return StubASREngine()


asr_session_manager = ASRSessionManager()
