from __future__ import annotations

import pytest
from pydantic import ValidationError

from libs.schemas import (
    BehaviorSignal,
    CredibilitySignal,
    EvidenceRef,
    ProbeResponse,
    ProbeSuggestion,
    QATurn,
    TranscriptSegment,
)


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


def test_evidence_ref_rejects_invalid_quote_time_ranges() -> None:
    ref = EvidenceRef(
        turn_id="turn-1",
        quote_start_ms=100,
        quote_end_ms=100,
        excerpt="回答片段",
    )

    assert ref.quote_start_ms == 100
    with pytest.raises(ValidationError):
        EvidenceRef(
            turn_id="turn-1",
            quote_start_ms=-1,
            quote_end_ms=100,
            excerpt="回答片段",
        )
    with pytest.raises(ValidationError):
        EvidenceRef(
            turn_id="turn-1",
            quote_start_ms=200,
            quote_end_ms=100,
            excerpt="回答片段",
        )


def test_probe_response_requires_one_to_three_suggestions() -> None:
    credibility = CredibilitySignal(
        level="vague",
        reason="缺少具体细节",
        drill_down_hint="追问本人负责部分",
    )
    suggestion = ProbeSuggestion(
        question="请讲清楚你本人负责哪一段？",
        target="验证项目真实性",
        competency="项目真实性",
        priority=1,
    )

    response = ProbeResponse(suggestions=[suggestion], credibility=credibility)

    assert response.suggestions[0].priority == 1
    with pytest.raises(ValidationError):
        ProbeResponse(suggestions=[], credibility=credibility)
    with pytest.raises(ValidationError):
        ProbeResponse(
            suggestions=[
                suggestion.model_copy(update={"priority": priority})
                for priority in [1, 2, 3, 3]
            ],
            credibility=credibility,
        )
