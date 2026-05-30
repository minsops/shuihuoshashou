from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from typing import Literal

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


def get_asr_engine() -> ASREngine:
    return StubASREngine()
