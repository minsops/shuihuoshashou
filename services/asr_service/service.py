from __future__ import annotations

import base64
from abc import ABC, abstractmethod

from libs.schemas import TranscriptSegment


class ASREngine(ABC):
    @abstractmethod
    async def transcribe_chunk(self, session_id: str, seq: int, audio_b64: str) -> TranscriptSegment:
        raise NotImplementedError


class StubASREngine(ASREngine):
    async def transcribe_chunk(self, session_id: str, seq: int, audio_b64: str) -> TranscriptSegment:
        try:
            raw = base64.b64decode(audio_b64).decode("utf-8")
        except Exception:
            raw = ""
        return TranscriptSegment(
            session_id=session_id,
            speaker="candidate" if seq % 2 else "interviewer",
            text=raw or "本地 ASR stub 收到音频片段，生产环境请替换为云 ASR 或 faster-whisper。",
            start_ms=seq * 1000,
            end_ms=seq * 1000 + 900,
            is_final=True,
            confidence=0.8,
        )


def get_asr_engine() -> ASREngine:
    return StubASREngine()

