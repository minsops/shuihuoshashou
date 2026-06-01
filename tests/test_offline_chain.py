from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

from libs.common.config import get_settings
from libs.common.database import connect, init_db, loads
from libs.common.events import event_bus
from libs.common.tasks import task_queue
from libs.schemas import (
    CandidateCreate,
    CompetencyItem,
    CompetencyModel,
    DimensionScore,
    EvidenceRef,
    InterviewCreate,
    InterviewContext,
    InterviewRecord,
    InterviewScore,
    InterviewStatus,
    JobCreate,
    JobRecord,
    ProbeRequest,
    QATurn,
    TranscriptSegment,
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
    should_probe,
)
from services.jd_kb_service.service import (
    _index_probe_patterns,
    _pgvector_literal,
    create_job,
    generate_competency_model,
    retrieve_job_probe_patterns,
)
from libs.common.prompts import load_prompt
from services.probe_service.service import fallback_probe, generate_probe
from services.scoring_service.service import score_interview


@pytest.fixture(autouse=True)
def clear_settings_cache_between_tests():
    yield
    get_settings.cache_clear()


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
    assert report.transcript
    assert report.transcript[0].answer == "我主要负责整体架构设计并推动项目落地最终取得显著提升"
    assert (tmp_path / "reports" / f"{interview.id}.html").exists()
    assert (tmp_path / "reports" / f"{interview.id}.transcript.json").exists()
    html = Path(report.html_path or "").read_text(encoding="utf-8")
    transcript_json = loads(Path(report.transcript_path or "").read_text(encoding="utf-8"))
    assert "data:image/png;base64" in html
    assert "亮点" in html
    assert "AIGC 察重" in html
    assert "疑似注水/模板化" in html
    assert "转写全文" in html
    assert report.transcript[0].answer in html
    assert report.artifact_uris["html"].startswith("file://")
    assert report.artifact_uris["pdf"].startswith("file://")
    assert report.artifact_uris["transcript"].startswith("file://")
    assert transcript_json[0]["answer"] == report.transcript[0].answer
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


def test_end_interview_can_queue_offline_scoring_without_running_it(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'async.db'}")
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
        ),
    )

    accepted = end_interview(interview.id, execute_inline=False)

    assert accepted.status == "queued"
    assert accepted.interview_id == interview.id
    assert accepted.task_name == "interview.offline_scoring"
    assert get_interview(interview.id).status == InterviewStatus.finished
    assert task_queue.history("interview.offline_scoring")[0].status == "queued"
    assert [topic for topic, _ in event_bus.history()][-2:] == [
        "interview.finished",
        "task.enqueued",
    ]


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
        "transcript": f"s3://reports-bucket/reports/{interview.id}.transcript.json",
    }
    assert Path(report.html_path or "").exists()
    assert Path(report.pdf_path or "").exists()
    assert Path(report.transcript_path or "").exists()


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


def test_generate_probe_uses_prompt_file(monkeypatch) -> None:
    messages = []

    class FakeClient:
        async def complete_json(self, sent_messages, schema, fallback):
            messages.extend(sent_messages)
            return fallback

    model = CompetencyModel(
        job_id="job-local",
        job_title="Backend",
        items=[
            CompetencyItem(name="项目真实性", description="验证项目经历", weight=1.0),
            CompetencyItem(name="工程深度", description="验证工程判断", weight=1.0),
            CompetencyItem(name="异常处理", description="验证故障经验", weight=1.0),
        ],
    )
    request = ProbeRequest(
        job_id="job-local",
        competency_model=model,
        recent_turns=[],
        latest_answer="我写了 FastAPI 编排、模型重试和 JSON 校验，因为线上有格式漂移。",
    )

    monkeypatch.setattr("services.probe_service.service.retrieve_job_probe_patterns", lambda *args, **kwargs: [])
    monkeypatch.setattr("services.probe_service.service.get_llm_client", lambda: FakeClient())

    __import__("asyncio").run(generate_probe(request))

    assert messages[0].role == "system"
    assert messages[0].content == load_prompt("probe_system.md")


def test_score_interview_uses_prompt_and_recomputes_total(monkeypatch) -> None:
    sent_messages = []
    model = generate_competency_model("job-local", "Backend", "Python FastAPI")
    turn = QATurn(
        question="讲项目",
        answer="我负责 FastAPI 接口优化，延迟降低 30%。",
        answer_start_ms=100,
        answer_end_ms=1200,
    )
    ctx = InterviewContext(
        session_id="session-local",
        job_id=model.job_id,
        candidate_id="candidate-local",
        competency_model=model,
        turns=[turn],
    )
    draft = InterviewScore(
        session_id=ctx.session_id,
        dimensions=[
            DimensionScore(
                dimension=item.name,
                score=90.0 if item.weight > 0 else 50.0,
                weight=999.0,
                evidence=[
                    EvidenceRef(
                        turn_id=turn.turn_id,
                        quote_start_ms=turn.answer_start_ms,
                        quote_end_ms=turn.answer_end_ms,
                        excerpt=turn.answer,
                    )
                ],
            )
            for item in model.items
        ],
        total_score=1.0,
        risk_notes=["LLM draft risk"],
        recommendation="no",
    )

    class FakeClient:
        def complete_json_sync(self, messages, schema, fallback):
            sent_messages.extend(messages)
            return draft

    monkeypatch.setattr("services.scoring_service.service.get_llm_client", lambda: FakeClient())

    score = score_interview(ctx, [])

    assert sent_messages[0].role == "system"
    assert sent_messages[0].content == load_prompt("scoring_system.md")
    assert score.total_score == 85.0
    assert score.recommendation == "yes"
    assert all(dimension.weight != 999.0 for dimension in score.dimensions)
    assert score.risk_notes == ["LLM draft risk"]


def test_jd_kb_retrieves_relevant_probe_patterns(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'kb.db'}")
    get_settings.cache_clear()
    init_db()
    job = create_job(JobCreate(title="LLM Backend", jd_text="Python FastAPI LLM 可靠性"))

    hits = retrieve_job_probe_patterns(job.id, "LLM 调用失败降级和 FastAPI 异常处理", limit=3)

    assert hits
    assert hits[0].score > 0
    assert any("LLM" in hit.pattern or "FastAPI" in hit.pattern for hit in hits)


def test_jd_kb_indexes_probe_pattern_embeddings(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'kb-vector.db'}")
    get_settings.cache_clear()
    init_db()
    job = create_job(JobCreate(title="LLM Backend", jd_text="Python FastAPI LLM 可靠性"))

    with connect() as conn:
        rows = conn.execute(
            "SELECT competency, pattern, embedding FROM probe_patterns WHERE job_id = ?",
            (job.id,),
        ).fetchall()

    hits = retrieve_job_probe_patterns(job.id, "模型调用失败时怎么降级", limit=2)

    assert rows
    assert all(loads(row["embedding"]) for row in rows)
    assert hits
    assert hits[0].score > 0


def test_jd_kb_pgvector_retrieval_uses_nearest_neighbor_sql(monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/app")
    monkeypatch.setenv("JD_VECTOR_BACKEND", "pgvector")
    get_settings.cache_clear()
    executed: list[tuple[str, tuple]] = []

    class Result:
        def fetchall(self):
            return [
                {
                    "competency": "专业能力深度",
                    "pattern": "请追问 LLM 调用、评估、成本、失败降级和安全边界。",
                    "score": 0.83,
                }
            ]

    class FakeConnection:
        def execute(self, query: str, params: tuple):
            executed.append((query, params))
            return Result()

    @contextmanager
    def fake_connect():
        yield FakeConnection()

    monkeypatch.setattr("services.jd_kb_service.service.init_db", lambda: None)
    monkeypatch.setattr("services.jd_kb_service.service.connect", fake_connect)

    hits = retrieve_job_probe_patterns("00000000-0000-0000-0000-000000000001", "LLM 降级", limit=3)

    assert hits[0].score == 0.83
    query, params = executed[0]
    assert "embedding_vector <=>" in query
    assert params[0].startswith("[")
    assert params[0] == params[2]
    assert params[3] == 3


def test_jd_kb_pgvector_index_writes_vector_column(monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/app")
    monkeypatch.setenv("JD_VECTOR_BACKEND", "pgvector")
    get_settings.cache_clear()
    inserted: list[tuple[str, tuple]] = []
    job = JobRecord(
        title="LLM Backend",
        jd_text="Python FastAPI LLM",
        competency_model=generate_competency_model(
            "00000000-0000-0000-0000-000000000001",
            "LLM Backend",
            "Python FastAPI LLM",
        ),
    )

    class FakeConnection:
        def execute(self, query: str, params: tuple):
            inserted.append((query, params))

    @contextmanager
    def fake_connect():
        yield FakeConnection()

    monkeypatch.setattr("services.jd_kb_service.service.connect", fake_connect)

    _index_probe_patterns(job)

    assert inserted
    query, params = inserted[0]
    assert "embedding_vector" in query
    assert "?::vector" in query
    assert params[5] == _pgvector_literal(loads(params[4]))


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


def test_aigc_detection_can_use_http_detector(monkeypatch) -> None:
    monkeypatch.setenv("AIGC_DETECTOR_PROVIDER", "http")
    monkeypatch.setenv("AIGC_DETECTOR_BASE_URL", "https://aigc.example.test/api")
    monkeypatch.setenv("AIGC_DETECTOR_API_PATH", "/v1/detect")
    monkeypatch.setenv("AIGC_DETECTOR_API_KEY", "secret")
    monkeypatch.setenv("AIGC_DETECTOR_AUTH_HEADER", "X-Detector-Key")
    monkeypatch.setenv("AIGC_DETECTOR_AUTH_SCHEME", "")
    get_settings.cache_clear()
    turn = QATurn(question="q", answer="我写了 FastAPI 编排和 JSON 校验。")

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://aigc.example.test/api/v1/detect"
        assert request.headers["X-Detector-Key"] == "secret"
        payload = __import__("json").loads(request.content)
        assert payload["turn_id"] == turn.turn_id
        assert payload["answer"] == turn.answer
        assert "local_template_similarity" in payload
        return httpx.Response(200, json={"ai_generated_prob": 0.91, "flagged": "false"})

    result = detect_interview([turn], transport=httpx.MockTransport(handler))[0]

    assert result.ai_generated_prob == 0.91
    assert result.flagged is False
    assert result.template_similarity >= 0.0


def test_aigc_http_detector_falls_back_on_failure(monkeypatch) -> None:
    monkeypatch.setenv("AIGC_DETECTOR_PROVIDER", "http")
    monkeypatch.setenv("AIGC_DETECTOR_BASE_URL", "https://aigc.example.test")
    get_settings.cache_clear()
    turn = QATurn(question="q", answer="我主要负责整体架构设计并推动项目落地最终取得显著提升")
    transport = httpx.MockTransport(lambda _: httpx.Response(500, json={"error": "boom"}))

    result = detect_interview([turn], transport=transport)[0]

    assert result.flagged is True
    assert result.template_similarity > 0.9


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


def test_should_probe_uses_configurable_thresholds(monkeypatch) -> None:
    monkeypatch.setenv("PROBE_MIN_ANSWER_CHARS", "5")
    monkeypatch.setenv("PROBE_MIN_INTERVAL_MS", "2500")
    get_settings.cache_clear()
    model = generate_competency_model("job-local", "Backend", "Python 服务端岗位")
    record = InterviewRecord(
        job_id=model.job_id,
        candidate_id="candidate-local",
        context=InterviewContext(
            session_id="session-local",
            job_id=model.job_id,
            candidate_id="candidate-local",
            competency_model=model,
        ),
    )
    record.context.turns.append(
        QATurn(question="q", answer="上一段回答", answer_start_ms=0, answer_end_ms=1000)
    )

    early = TranscriptSegment(
        session_id=record.id,
        speaker="candidate",
        text="我负责项目里的接口优化",
        start_ms=3000,
        end_ms=3200,
        is_final=True,
        confidence=0.9,
    )
    late = early.model_copy(update={"start_ms": 3600})

    assert should_probe(early, record) is False
    assert should_probe(late, record) is True


def test_should_probe_requires_drill_down_topic_by_default(monkeypatch) -> None:
    monkeypatch.delenv("PROBE_REQUIRE_TOPIC_MATCH", raising=False)
    monkeypatch.setenv("PROBE_MIN_ANSWER_CHARS", "5")
    get_settings.cache_clear()
    model = generate_competency_model("job-local", "Backend", "Python 服务端岗位")
    record = InterviewRecord(
        job_id=model.job_id,
        candidate_id="candidate-local",
        context=InterviewContext(
            session_id="session-local",
            job_id=model.job_id,
            candidate_id="candidate-local",
            competency_model=model,
        ),
    )
    casual = TranscriptSegment(
        session_id=record.id,
        speaker="candidate",
        text="今天状态还可以，整体感觉比较顺利。",
        start_ms=0,
        end_ms=1000,
        is_final=True,
        confidence=0.9,
    )
    project = casual.model_copy(update={"text": "我负责项目里的 FastAPI 架构优化。"})

    assert should_probe(casual, record) is False
    assert should_probe(project, record) is True
