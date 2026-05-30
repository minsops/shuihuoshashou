from __future__ import annotations

from pathlib import Path

from libs.common.config import get_settings
from libs.common.database import init_db
from libs.common.events import event_bus
from libs.common.tasks import task_queue
from libs.schemas import (
    CandidateCreate,
    InterviewCreate,
    InterviewStatus,
    JobCreate,
    ProbeRequest,
    QATurn,
)
from services.aigc_detect_service.service import detect_interview, load_templates
from services.interview_orchestrator.consistency import detect_consistency, extract_fact_claim
from services.interview_orchestrator.service import (
    add_turn,
    create_candidate,
    create_interview,
    end_interview,
    finish_interview,
    get_interview,
    list_turns,
    run_offline_scoring_task,
)
from services.jd_kb_service.service import create_job, retrieve_job_probe_patterns
from services.probe_service.service import fallback_probe


def test_offline_interview_chain(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("REPORT_DIR", str(tmp_path / "reports"))
    get_settings.cache_clear()
    event_bus.reset()
    task_queue.reset()
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
    assert "data:image/png;base64" in Path(report.html_path or "").read_text(encoding="utf-8")
    assert report.artifact_uris["html"].startswith("file://")
    assert report.artifact_uris["pdf"].startswith("file://")
    assert get_interview(interview.id).status == InterviewStatus.reported
    assert [turn.turn_id for turn in list_turns(interview.id)]
    assert [topic for topic, _ in event_bus.history()] == [
        "qa_turn.created",
        "interview.finished",
        "task.enqueued",
        "interview.scoring_started",
        "interview.reported",
        "task.completed",
    ]
    tasks = task_queue.history("interview.offline_scoring")
    assert len(tasks) == 1
    assert tasks[0].status == "completed"


def test_offline_pipeline_exposes_scoring_state(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'pipeline.db'}")
    monkeypatch.setenv("REPORT_DIR", str(tmp_path / "reports"))
    get_settings.cache_clear()
    event_bus.reset()
    task_queue.reset()
    init_db()
    job = create_job(JobCreate(title="Backend", jd_text="Python FastAPI"))
    candidate = create_candidate(CandidateCreate(name="Grace"))
    interview = create_interview(InterviewCreate(job_id=job.id, candidate_id=candidate.id))
    add_turn(
        interview.id,
        QATurn(
            question="讲项目",
            answer="我写了 FastAPI 编排、模型重试和 JSON 校验，因为线上有格式漂移。",
            answer_start_ms=100,
            answer_end_ms=1200,
        ),
    )

    finished = finish_interview(interview.id)
    assert finished.status == InterviewStatus.finished
    assert get_interview(interview.id).status == InterviewStatus.finished

    report = run_offline_scoring_task(interview.id)

    assert report.interview_id == interview.id
    assert get_interview(interview.id).status == InterviewStatus.reported
    topics = [topic for topic, _ in event_bus.history()]
    assert "interview.scoring_started" in topics
    assert "interview.reported" in topics


def test_report_artifact_uris_use_object_storage_when_configured(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'object.db'}")
    monkeypatch.setenv("REPORT_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("OBJECT_STORAGE_ENDPOINT", "http://minio:9000")
    monkeypatch.setenv("OBJECT_STORAGE_BUCKET", "reports-bucket")
    get_settings.cache_clear()
    event_bus.reset()
    task_queue.reset()
    init_db()
    job = create_job(JobCreate(title="Backend", jd_text="Python FastAPI"))
    candidate = create_candidate(CandidateCreate(name="Lin"))
    interview = create_interview(InterviewCreate(job_id=job.id, candidate_id=candidate.id))
    add_turn(
        interview.id,
        QATurn(
            question="讲项目",
            answer="我写了 FastAPI 编排、模型重试和 JSON 校验，因为线上有格式漂移。",
        ),
    )

    report = end_interview(interview.id)

    assert report.artifact_uris == {
        "html": f"s3://reports-bucket/reports/{interview.id}.html",
        "pdf": f"s3://reports-bucket/reports/{interview.id}.pdf",
    }
    assert Path(report.html_path or "").exists()
    assert Path(report.pdf_path or "").exists()


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


def test_jd_kb_retrieves_relevant_probe_patterns(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'kb.db'}")
    get_settings.cache_clear()
    init_db()
    job = create_job(JobCreate(title="LLM Backend", jd_text="Python FastAPI LLM 可靠性"))

    hits = retrieve_job_probe_patterns(job.id, "LLM 调用失败降级和 FastAPI 异常处理", limit=3)

    assert hits
    assert hits[0].score > 0
    assert any("LLM" in hit.pattern or "FastAPI" in hit.pattern for hit in hits)


def test_probe_fallback_uses_retrieved_patterns() -> None:
    job = create_job(JobCreate(title="LLM Backend", jd_text="Python FastAPI LLM 可靠性"))
    response = fallback_probe(
        ProbeRequest(
            job_id=job.id,
            competency_model=job.competency_model,
            recent_turns=[],
            latest_answer="我主要负责 LLM 调用和 FastAPI 优化，效果比较好。",
        )
    )

    assert any("LLM" in suggestion.question or "FastAPI" in suggestion.question for suggestion in response.suggestions)


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
    assert results[0].template_similarity > 0.9


def test_aigc_detection_uses_template_corpus_for_paraphrase() -> None:
    results = detect_interview(
        [
            QATurn(
                question="q",
                answer="首先我分析业务痛点，然后制定技术方案，最后推动落地并持续优化。",
            )
        ]
    )
    assert len(load_templates()) >= 5
    assert results[0].matched_template
    assert results[0].template_similarity >= 0.45


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
