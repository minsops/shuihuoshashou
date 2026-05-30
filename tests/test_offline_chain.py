from __future__ import annotations

from pathlib import Path

from libs.common.config import get_settings
from libs.common.database import init_db
from libs.schemas import CandidateCreate, InterviewCreate, JobCreate, ProbeRequest, QATurn
from services.aigc_detect_service.service import detect_interview
from services.interview_orchestrator.consistency import detect_consistency, extract_fact_claim
from services.interview_orchestrator.service import (
    add_turn,
    create_candidate,
    create_interview,
    end_interview,
)
from services.jd_kb_service.service import create_job
from services.probe_service.service import fallback_probe


def test_offline_interview_chain(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("REPORT_DIR", str(tmp_path / "reports"))
    get_settings.cache_clear()
    init_db()
    job = create_job(JobCreate(title="AI Engineer", jd_text="Python FastAPI LLM"))
    candidate = create_candidate(CandidateCreate(name="Ada"))
    interview = create_interview(InterviewCreate(job_id=job.id, candidate_id=candidate.id))
    add_turn(
        interview.id,
        QATurn(
            question="Tell me about a project.",
            answer="我主要负责整体架构设计并推动项目落地最终取得显著提升",
            answer_start_ms=0,
            answer_end_ms=1000,
        ),
    )
    report = end_interview(interview.id)
    assert report.score.total_score > 0
    assert report.aigc_results
    assert (tmp_path / "reports" / f"{interview.id}.html").exists()


def test_probe_fallback_returns_three_suggestions() -> None:
    job = create_job(JobCreate(title="Backend", jd_text="Python"))
    request = ProbeRequest(
        job_id=job.id,
        competency_model=job.competency_model,
        recent_turns=[],
        latest_answer="我主要负责优化，做了很多事情，效果比较好。",
    )
    response = fallback_probe(request)
    assert len(response.suggestions) == 3
    assert response.credibility.level in {"vague", "suspicious"}


def test_aigc_detection_flags_template() -> None:
    results = detect_interview(
        [
            QATurn(
                question="q",
                answer="我主要负责整体架构设计并推动项目落地最终取得显著提升",
            )
        ]
    )
    assert results[0].flagged


def test_fact_claim_extraction() -> None:
    turn = QATurn(
        question="q",
        answer="我独立负责 FastAPI 编排和重试，接口耗时降低 30%。",
    )
    claim = extract_fact_claim(turn)
    assert claim.contribution_scope == "solo"
    assert "FastAPI" in claim.technologies
    assert "重试" in claim.responsibilities
    assert claim.metrics


def test_consistency_detects_contribution_conflict() -> None:
    turns = [
        QATurn(question="q1", answer="这个项目是我独立完成的，我负责整体架构。"),
        QATurn(question="q2", answer="其实核心链路是团队负责，同事负责主要实现。"),
    ]
    flags = detect_consistency(turns)
    assert flags
    assert flags[0].severity == "high"
