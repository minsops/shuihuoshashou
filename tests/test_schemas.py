from __future__ import annotations

import pytest
from pydantic import ValidationError

from libs.schemas import BehaviorSignal, QATurn, TranscriptSegment


def test_behavior_signal_forbids_compliance_sensitive_extra_fields() -> None:
    signal = BehaviorSignal.model_validate(
        {
            "turn_id": "turn-1",
            "fluency": 0.8,
            "hesitation": 0.2,
            "evasiveness_hint": False,
        }
    )

    assert signal.model_dump() == {
        "turn_id": "turn-1",
        "fluency": 0.8,
        "hesitation": 0.2,
        "evasiveness_hint": False,
    }

    with pytest.raises(ValidationError):
        BehaviorSignal.model_validate(
            {
                "turn_id": "turn-1",
                "fluency": 0.8,
                "hesitation": 0.2,
                "evasiveness_hint": False,
                "personality": "confident",
                "emotion": "nervous",
                "reliability": 0.4,
            }
        )


def test_transcript_segment_rejects_invalid_time_ranges() -> None:
    segment = TranscriptSegment(
        session_id="session-1",
        speaker="candidate",
        text="回答",
        start_ms=100,
        end_ms=100,
        is_final=True,
        confidence=0.8,
    )

    assert segment.start_ms == 100
    with pytest.raises(ValidationError):
        TranscriptSegment(
            session_id="session-1",
            speaker="candidate",
            text="回答",
            start_ms=-1,
            end_ms=100,
            is_final=True,
            confidence=0.8,
        )
    with pytest.raises(ValidationError):
        TranscriptSegment(
            session_id="session-1",
            speaker="candidate",
            text="回答",
            start_ms=200,
            end_ms=100,
            is_final=True,
            confidence=0.8,
        )


def test_qa_turn_rejects_invalid_answer_time_ranges() -> None:
    turn = QATurn(question="q", answer="a", answer_start_ms=0, answer_end_ms=10)

    assert turn.answer_end_ms == 10
    with pytest.raises(ValidationError):
        QATurn(question="q", answer="a", answer_start_ms=-1, answer_end_ms=10)
    with pytest.raises(ValidationError):
        QATurn(question="q", answer="a", answer_start_ms=20, answer_end_ms=10)
