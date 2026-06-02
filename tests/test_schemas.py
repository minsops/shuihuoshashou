from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from libs.schemas import (
    AIGCDetectRequest,
    AIGCResult,
    BehaviorSignal,
    CandidateCreate,
    ConsentCreate,
    ConsentRecord,
    ConsistencyFlag,
    CredibilitySignal,
    CompetencyItem,
    CompetencyModel,
    DimensionScore,
    EvidenceRef,
    FactClaim,
    InterviewCreate,
    InterviewContext,
    InterviewRecord,
    InterviewScore,
    InterviewStatus,
    JobCreate,
    OfflineTaskAccepted,
    ProbePatternHit,
    ProbeResponse,
    ProbeRequest,
    ProbeSuggestion,
    QATurn,
    Report,
    ReportBuildRequest,
    ScoringRequest,
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


def test_qa_turn_rejects_blank_question_or_answer() -> None:
    probe_turn = QATurn(
        question="追问",
        question_source="ai_probe",
        answer="回答",
        probe_target="验证项目真实性",
    )

    assert probe_turn.probe_target == "验证项目真实性"
    with pytest.raises(ValidationError):
        QATurn(question=" ", answer="a")
    with pytest.raises(ValidationError):
        QATurn(question="q", answer=" ")
    with pytest.raises(ValidationError):
        QATurn(question="q", answer="a", probe_target=" ")
    with pytest.raises(ValidationError):
        QATurn(question="q", question_source="ai_probe", answer="a")


def test_job_candidate_and_competency_payloads_reject_blank_required_text() -> None:
    with pytest.raises(ValidationError):
        JobCreate(title=" ", jd_text="Python FastAPI")
    with pytest.raises(ValidationError):
        JobCreate(title="Backend", jd_text=" ")
    with pytest.raises(ValidationError):
        CandidateCreate(name=" ")
    with pytest.raises(ValidationError):
        CompetencyItem(name=" ", description="验证工程能力", weight=1.0)
    with pytest.raises(ValidationError):
        CompetencyItem(name="工程深度", description=" ", weight=1.0)
    with pytest.raises(ValidationError):
        CompetencyItem(name="工程深度", description="验证工程能力", weight=float("nan"))
    with pytest.raises(ValidationError):
        CompetencyItem(name="工程深度", description="验证工程能力", weight=float("inf"))
    with pytest.raises(ValidationError):
        CompetencyItem(
            name="工程深度",
            description="验证工程能力",
            probe_patterns=["请讲一个具体案例。", " "],
            weight=1.0,
        )
    with pytest.raises(ValidationError):
        CompetencyModel(
            job_id="job-1",
            job_title=" ",
            items=[
                CompetencyItem(
                    name="工程深度",
                    description="验证工程能力",
                    weight=1.0,
                )
            ],
        )


def test_shared_contract_models_reject_blank_identifiers() -> None:
    competency = CompetencyItem(name="项目真实性", description="验证项目经历", weight=1.0)
    model = CompetencyModel(job_id="job-1", job_title="Backend", items=[competency])
    evidence = EvidenceRef(turn_id="turn-1", quote_start_ms=0, quote_end_ms=10, excerpt="回答")
    dimension = DimensionScore(
        dimension="项目真实性",
        score=80.0,
        weight=1.0,
        evidence=[evidence],
    )
    score = InterviewScore(
        session_id="session-1",
        dimensions=[dimension],
        total_score=80.0,
        recommendation="yes",
    )
    aigc = AIGCResult(turn_id="turn-1", ai_generated_prob=0.2, template_similarity=0.1)

    with pytest.raises(ValidationError):
        CompetencyModel(job_id=" ", job_title="Backend", items=[competency])
    with pytest.raises(ValidationError):
        ProbePatternHit(job_id=" ", competency="项目真实性", pattern="请讲具体项目。", score=1.0)
    with pytest.raises(ValidationError):
        ProbePatternHit(
            job_id="job-1",
            competency="项目真实性",
            pattern="请讲具体项目。",
            score=float("inf"),
        )
    with pytest.raises(ValidationError):
        ConsistencyFlag(turn_id_a=" ", turn_id_b="turn-2", description="矛盾", severity="high")
    with pytest.raises(ValidationError):
        ConsistencyFlag(
            turn_id_a="turn-1",
            turn_id_b="turn-1",
            description="同一回答不能构成前后一致性冲突",
            severity="high",
        )
    with pytest.raises(ValidationError):
        FactClaim(turn_id=" ")
    with pytest.raises(ValidationError):
        FactClaim(turn_id="turn-1", responsibilities=["架构", " "])
    with pytest.raises(ValidationError):
        FactClaim(turn_id="turn-1", technologies=["FastAPI", " "])
    with pytest.raises(ValidationError):
        FactClaim(turn_id="turn-1", metrics=["30%", " "])
    with pytest.raises(ValidationError):
        QATurn(turn_id=" ", question="q", answer="a")
    with pytest.raises(ValidationError):
        ProbeRequest(job_id=" ", competency_model=model, recent_turns=[], latest_answer="回答")
    with pytest.raises(ValidationError):
        EvidenceRef(turn_id=" ", quote_start_ms=0, quote_end_ms=10, excerpt="回答")
    with pytest.raises(ValidationError):
        InterviewScore(
            session_id=" ",
            dimensions=[dimension],
            total_score=80.0,
            recommendation="yes",
        )
    with pytest.raises(ValidationError):
        AIGCResult(turn_id=" ", ai_generated_prob=0.2, template_similarity=0.1)
    with pytest.raises(ValidationError):
        AIGCResult(
            turn_id="turn-1",
            ai_generated_prob=0.2,
            template_similarity=0.1,
            matched_template=" ",
        )
    with pytest.raises(ValidationError):
        BehaviorSignal(turn_id=" ", fluency=0.8, hesitation=0.1, evasiveness_hint=False)
    with pytest.raises(ValidationError):
        ConsentCreate(candidate_id=" ")
    with pytest.raises(ValidationError):
        InterviewCreate(job_id=" ", candidate_id="candidate-1")
    with pytest.raises(ValidationError):
        InterviewCreate(job_id="job-1", candidate_id=" ")
    with pytest.raises(ValidationError):
        OfflineTaskAccepted(interview_id=" ", task_id="task-1", task_name="task")
    with pytest.raises(ValidationError):
        Report(
            interview_id=" ",
            score=score,
            aigc_results=[aigc],
            consistency_flags=[],
            summary="报告摘要",
        )


def test_aigc_detect_request_requires_candidate_turns() -> None:
    turn = QATurn(turn_id="turn-1", question="q", answer="a")

    request = AIGCDetectRequest(turns=[turn])

    assert request.turns[0].turn_id == "turn-1"
    with pytest.raises(ValidationError):
        AIGCDetectRequest(turns=[])
    with pytest.raises(ValidationError):
        AIGCDetectRequest(turns=[turn, turn])


def test_interview_context_rejects_duplicate_turn_ids() -> None:
    competency = CompetencyItem(name="项目真实性", description="验证项目经历", weight=1.0)
    model = CompetencyModel(job_id="job-1", job_title="Backend", items=[competency])
    turn = QATurn(turn_id="turn-1", question="q", answer="a")

    context = InterviewContext(
        session_id="session-1",
        job_id="job-1",
        candidate_id="candidate-1",
        competency_model=model,
        turns=[turn],
    )

    assert context.turns[0].turn_id == "turn-1"
    with pytest.raises(ValidationError):
        InterviewContext(
            session_id="session-1",
            job_id="job-1",
            candidate_id="candidate-1",
            competency_model=model,
            turns=[turn, turn],
        )


def test_interview_timestamps_must_be_monotonic() -> None:
    started_at = datetime(2026, 6, 2, 10, 0, tzinfo=UTC)
    competency = CompetencyItem(name="项目真实性", description="验证项目经历", weight=1.0)
    model = CompetencyModel(job_id="job-1", job_title="Backend", items=[competency])
    context = InterviewContext(
        session_id="session-1",
        job_id="job-1",
        candidate_id="candidate-1",
        competency_model=model,
        started_at=started_at,
        ended_at=started_at,
    )
    record = InterviewRecord(
        job_id="job-1",
        candidate_id="candidate-1",
        status=InterviewStatus.finished,
        context=context,
        started_at=started_at,
        ended_at=started_at,
    )

    assert context.ended_at == started_at
    assert record.ended_at == started_at
    with pytest.raises(ValidationError):
        InterviewContext(
            session_id="session-1",
            job_id="job-1",
            candidate_id="candidate-1",
            competency_model=model,
            started_at=started_at,
            ended_at=started_at - timedelta(seconds=1),
        )
    with pytest.raises(ValidationError):
        InterviewRecord(
            job_id="job-1",
            candidate_id="candidate-1",
            status=InterviewStatus.finished,
            context=context,
            started_at=started_at,
            ended_at=started_at - timedelta(seconds=1),
        )


def test_interview_record_status_requires_matching_timestamps() -> None:
    started_at = datetime(2026, 6, 2, 10, 0, tzinfo=UTC)
    ended_at = started_at + timedelta(minutes=30)
    competency = CompetencyItem(name="项目真实性", description="验证项目经历", weight=1.0)
    model = CompetencyModel(job_id="job-1", job_title="Backend", items=[competency])
    context = InterviewContext(
        session_id="session-1",
        job_id="job-1",
        candidate_id="candidate-1",
        competency_model=model,
    )

    in_progress = InterviewRecord(
        job_id="job-1",
        candidate_id="candidate-1",
        status=InterviewStatus.in_progress,
        context=context,
        started_at=started_at,
    )
    reported = InterviewRecord(
        job_id="job-1",
        candidate_id="candidate-1",
        status=InterviewStatus.reported,
        context=context,
        started_at=started_at,
        ended_at=ended_at,
    )

    assert in_progress.ended_at is None
    assert reported.ended_at == ended_at
    with pytest.raises(ValidationError):
        InterviewRecord(
            job_id="job-1",
            candidate_id="candidate-1",
            context=context,
            started_at=started_at,
        )
    with pytest.raises(ValidationError):
        InterviewRecord(
            job_id="job-1",
            candidate_id="candidate-1",
            status=InterviewStatus.in_progress,
            context=context,
        )
    with pytest.raises(ValidationError):
        InterviewRecord(
            job_id="job-1",
            candidate_id="candidate-1",
            status=InterviewStatus.in_progress,
            context=context,
            started_at=started_at,
            ended_at=ended_at,
        )
    with pytest.raises(ValidationError):
        InterviewRecord(
            job_id="job-1",
            candidate_id="candidate-1",
            status=InterviewStatus.finished,
            context=context,
            started_at=started_at,
        )


def test_consent_record_rejects_revocation_before_grant() -> None:
    granted_at = datetime(2026, 6, 2, 10, 0, tzinfo=UTC)
    record = ConsentRecord(
        candidate_id="candidate-1",
        granted=True,
        granted_at=granted_at,
        revoked_at=granted_at,
    )

    assert record.revoked_at == granted_at
    with pytest.raises(ValidationError):
        ConsentRecord(
            candidate_id="candidate-1",
            granted=True,
            granted_at=granted_at,
            revoked_at=granted_at - timedelta(seconds=1),
        )


def test_scoring_and_report_requests_require_aigc_results() -> None:
    competency = CompetencyItem(name="项目真实性", description="验证项目经历", weight=1.0)
    model = CompetencyModel(job_id="job-1", job_title="Backend", items=[competency])
    turn = QATurn(turn_id="turn-1", question="q", answer="a")
    context = InterviewContext(
        session_id="session-1",
        job_id="job-1",
        candidate_id="candidate-1",
        competency_model=model,
        turns=[turn],
    )
    score = InterviewScore(
        session_id="session-1",
        dimensions=[
            DimensionScore(
                dimension="项目真实性",
                score=80.0,
                weight=1.0,
                evidence=[
                    EvidenceRef(
                        turn_id="turn-1",
                        quote_start_ms=0,
                        quote_end_ms=0,
                        excerpt="a",
                    )
                ],
            )
        ],
        total_score=80.0,
        recommendation="yes",
    )

    with pytest.raises(ValidationError):
        ScoringRequest(context=context, aigc_results=[])
    with pytest.raises(ValidationError):
        ReportBuildRequest(context=context, score=score, aigc_results=[])


def test_report_requires_aigc_results_and_summary() -> None:
    evidence = EvidenceRef(
        turn_id="turn-1",
        quote_start_ms=0,
        quote_end_ms=10,
        excerpt="回答",
    )
    score = InterviewScore(
        session_id="session-1",
        dimensions=[
            DimensionScore(
                dimension="项目真实性",
                score=80.0,
                weight=1.0,
                evidence=[evidence],
            )
        ],
        total_score=80.0,
        recommendation="yes",
    )
    aigc = AIGCResult(turn_id="turn-1", ai_generated_prob=0.2, template_similarity=0.1)

    report = Report(
        interview_id="interview-1",
        score=score,
        aigc_results=[aigc],
        consistency_flags=[],
        summary="报告摘要",
    )

    assert report.aigc_results[0].turn_id == "turn-1"
    with pytest.raises(ValidationError):
        Report(
            interview_id="interview-1",
            score=score,
            aigc_results=[],
            consistency_flags=[],
            summary="报告摘要",
        )
    with pytest.raises(ValidationError):
        Report(
            interview_id="interview-1",
            score=score,
            aigc_results=[aigc],
            consistency_flags=[],
            summary=" ",
        )
    with pytest.raises(ValidationError):
        Report(
            interview_id="interview-1",
            score=score,
            aigc_results=[aigc],
            consistency_flags=[],
            summary="报告摘要",
            pdf_path=" ",
        )
    with pytest.raises(ValidationError):
        Report(
            interview_id="interview-1",
            score=score,
            aigc_results=[aigc],
            consistency_flags=[],
            summary="报告摘要",
            artifact_uris={"pdf": " "},
        )
    with pytest.raises(ValidationError):
        Report(
            interview_id="interview-1",
            score=score,
            aigc_results=[aigc],
            consistency_flags=[],
            summary="报告摘要",
            artifact_uris={" ": "file:///tmp/report.pdf"},
        )


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
    with pytest.raises(ValidationError):
        EvidenceRef(
            turn_id="turn-1",
            quote_start_ms=0,
            quote_end_ms=100,
            excerpt=" ",
        )


def test_dimension_scores_require_evidence_and_score_requires_dimensions() -> None:
    evidence = EvidenceRef(
        turn_id="turn-1",
        quote_start_ms=0,
        quote_end_ms=100,
        excerpt="回答片段",
    )
    dimension = DimensionScore(
        dimension="项目真实性",
        score=80.0,
        weight=0.25,
        evidence=[evidence],
    )

    assert dimension.evidence[0].turn_id == "turn-1"
    with pytest.raises(ValidationError):
        DimensionScore(
            dimension="项目真实性",
            score=80.0,
            weight=0.25,
            evidence=[],
        )
    with pytest.raises(ValidationError):
        DimensionScore(
            dimension="项目真实性",
            score=80.0,
            weight=float("nan"),
            evidence=[evidence],
        )
    with pytest.raises(ValidationError):
        DimensionScore(
            dimension="项目真实性",
            score=80.0,
            weight=float("inf"),
            evidence=[evidence],
        )
    with pytest.raises(ValidationError):
        InterviewScore(
            session_id="session-1",
            dimensions=[],
            total_score=80.0,
            recommendation="yes",
        )
    with pytest.raises(ValidationError):
        InterviewScore(
            session_id="session-1",
            dimensions=[dimension],
            total_score=80.0,
            risk_notes=[" "],
            recommendation="yes",
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


def test_probe_contract_rejects_blank_answer_and_card_text() -> None:
    competency = CompetencyItem(name="项目真实性", description="验证项目经历", weight=1.0)
    model = CompetencyModel(job_id="job-1", job_title="Backend", items=[competency])

    with pytest.raises(ValidationError):
        ProbeRequest(
            job_id="job-1",
            competency_model=model,
            recent_turns=[],
            latest_answer=" ",
        )
    with pytest.raises(ValidationError):
        ProbeSuggestion(
            question=" ",
            target="验证项目真实性",
            competency="项目真实性",
            priority=1,
        )
    with pytest.raises(ValidationError):
        ProbeSuggestion(
            question="请讲清楚你本人负责哪一段？",
            target=" ",
            competency="项目真实性",
            priority=1,
        )
    with pytest.raises(ValidationError):
        CredibilitySignal(level="vague", reason=" ", drill_down_hint="追问本人负责部分")
    with pytest.raises(ValidationError):
        CredibilitySignal(level="vague", reason="缺少具体细节", drill_down_hint=" ")
