from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from libs.common.config import get_settings
from libs.common.events import event_bus
from libs.common.observability import metrics_registry, reset_rate_limiters
from libs.common.tasks import task_queue
from services.asr_service.service import asr_session_manager
from services.gateway.app import app


def _client(
    tmp_path: Path,
    monkeypatch,
    *,
    rate_limit_enabled: bool = False,
    rate_limit_requests_per_minute: int = 120,
    gateway_api_key: str = "",
    signal_enabled: bool = False,
) -> TestClient:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'api.db'}")
    monkeypatch.setenv("REPORT_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("SIGNAL_ENABLED", str(signal_enabled).lower())
    monkeypatch.setenv("RATE_LIMIT_ENABLED", str(rate_limit_enabled).lower())
    monkeypatch.setenv("RATE_LIMIT_REQUESTS_PER_MINUTE", str(rate_limit_requests_per_minute))
    monkeypatch.setenv("GATEWAY_API_KEY", gateway_api_key)
    monkeypatch.setenv("OFFLINE_TASK_BACKEND", "local")
    get_settings.cache_clear()
    event_bus.reset()
    metrics_registry.reset()
    reset_rate_limiters()
    asr_session_manager.reset()
    task_queue.reset()
    return TestClient(app)


def test_gateway_offline_report_flow(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)

    job = client.post(
        "/api/jobs",
        json={"title": "AI 后端工程师", "jd_text": "Python FastAPI LLM 可靠性"},
    ).json()
    candidate = client.post(
        "/api/candidates",
        json={"name": "Candidate", "resume_text": "AI backend"},
    ).json()
    interview = client.post(
        "/api/interviews",
        json={"job_id": job["id"], "candidate_id": candidate["id"]},
    ).json()
    client.post(
        f"/api/interviews/{interview['id']}/turns",
        json={
            "question": "讲一个项目",
            "answer": "我主要负责整体架构设计并推动项目落地最终取得显著提升",
            "answer_start_ms": 0,
            "answer_end_ms": 1000,
        },
    )

    report = client.post(f"/api/interviews/{interview['id']}/end").json()
    assert report["score"]["total_score"] > 0
    assert client.get(f"/api/interviews/{interview['id']}/report").status_code == 200
    assert client.get(f"/api/interviews/{interview['id']}/report.html").status_code == 200
    report_json = client.get(f"/api/interviews/{interview['id']}/report.json")
    assert report_json.status_code == 200
    assert report_json.headers["content-type"].startswith("application/json")
    assert report_json.json()["interview_id"] == interview["id"]
    assert report_json.json()["artifact_uris"]["json"].startswith("file://")
    transcript = client.get(f"/api/interviews/{interview['id']}/report.transcript.json")
    assert transcript.status_code == 200
    assert transcript.headers["content-type"].startswith("application/json")
    assert transcript.json()[0]["answer"] == "我主要负责整体架构设计并推动项目落地最终取得显著提升"
    pdf = client.get(f"/api/interviews/{interview['id']}/report.pdf")
    assert pdf.status_code == 200
    assert pdf.headers["content-type"] == "application/pdf"
    assert pdf.content.startswith(b"%PDF")


def test_gateway_report_pdf_returns_404_when_artifact_missing(
    tmp_path: Path, monkeypatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    job = client.post("/api/jobs", json={"title": "Backend", "jd_text": "Python"}).json()
    candidate = client.post("/api/candidates", json={"name": "Candidate"}).json()
    interview = client.post(
        "/api/interviews",
        json={"job_id": job["id"], "candidate_id": candidate["id"]},
    ).json()
    client.post(
        f"/api/interviews/{interview['id']}/turns",
        json={"question": "讲项目", "answer": "我写了 FastAPI 编排。"},
    )
    report = client.post(f"/api/interviews/{interview['id']}/end").json()
    Path(report["pdf_path"]).unlink()

    missing = client.get(f"/api/interviews/{interview['id']}/report.pdf")

    assert missing.status_code == 404
    assert "report pdf not found" in missing.text


def test_gateway_report_transcript_returns_404_when_artifact_missing(
    tmp_path: Path, monkeypatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    job = client.post("/api/jobs", json={"title": "Backend", "jd_text": "Python"}).json()
    candidate = client.post("/api/candidates", json={"name": "Candidate"}).json()
    interview = client.post(
        "/api/interviews",
        json={"job_id": job["id"], "candidate_id": candidate["id"]},
    ).json()
    client.post(
        f"/api/interviews/{interview['id']}/turns",
        json={"question": "讲项目", "answer": "我写了 FastAPI 编排。"},
    )
    report = client.post(f"/api/interviews/{interview['id']}/end").json()
    Path(report["transcript_path"]).unlink()

    missing = client.get(f"/api/interviews/{interview['id']}/report.transcript.json")

    assert missing.status_code == 404
    assert "report transcript not found" in missing.text


def test_gateway_serves_demo_ui(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    response = client.get("/")
    assert response.status_code == 200
    assert "水货杀手" in response.text
    assert "/api/offline/evaluate" in response.text
    assert "实时追问" in response.text
    assert "/ws/interview/" in response.text
    assert "asr_warning" in response.text


def test_gateway_config_status_hides_secrets(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "super-secret")
    client = _client(tmp_path, monkeypatch)
    response = client.get("/api/config/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["database_url"].startswith("sqlite:///")
    assert payload["otel_exporter_otlp_configured"] is False
    assert payload["otel_service_name"] == "shuihuo-killer-gateway"
    assert payload["llm_api_key_configured"] is True
    assert payload["llm_max_retries"] == 1
    assert payload["llm_rate_limit_enabled"] is False
    assert payload["llm_rate_limit_requests_per_minute"] == 60
    assert payload["gateway_auth_enabled"] is False
    assert payload["asr_provider"] == "stub"
    assert payload["asr_base_url_configured"] is False
    assert payload["asr_api_path"] == "/transcribe"
    assert payload["asr_api_key_configured"] is False
    assert payload["asr_text_path"] == "text"
    assert payload["asr_speaker_path"] == "speaker"
    assert payload["asr_is_final_path"] == "is_final"
    assert payload["asr_confidence_path"] == "confidence"
    assert payload["asr_timeout_seconds"] == 30
    assert payload["asr_channel_diarization_configured"] is True
    assert payload["probe_min_answer_chars"] == 20
    assert payload["probe_min_interval_ms"] == 1000
    assert payload["probe_require_topic_match"] is True
    assert payload["probe_topic_keywords_configured"] is True
    assert payload["speaker_diarization_provider"] == "local"
    assert payload["speaker_diarization_base_url_configured"] is False
    assert payload["speaker_diarization_api_key_configured"] is False
    assert payload["speaker_diarization_speaker_path"] == "speaker"
    assert payload["speaker_diarization_timeout_seconds"] == 10
    assert payload["aigc_detector_provider"] == "local"
    assert payload["aigc_detector_base_url_configured"] is False
    assert payload["aigc_detector_api_path"] == "/detect"
    assert payload["aigc_detector_api_key_configured"] is False
    assert payload["aigc_detector_probability_path"] == "ai_generated_prob"
    assert payload["aigc_detector_flagged_path"] == "flagged"
    assert payload["aigc_detector_timeout_seconds"] == 10
    assert payload["aigc_ai_prob_threshold"] == 0.65
    assert payload["aigc_template_similarity_threshold"] == 0.45
    assert payload["rate_limit_enabled"] is False
    assert payload["rate_limit_backend"] == "local"
    assert payload["rate_limit_requests_per_minute"] == 120
    assert payload["redis_rate_limit_prefix"] == "shuihuo:rate_limit"
    assert payload["offline_task_backend"] == "local"
    assert payload["offline_task_execution"] == "sync"
    assert payload["celery_broker_configured"] is True
    assert payload["celery_result_backend_configured"] is True
    assert payload["celery_task_queue"] == "shuihuo-offline"
    assert payload["redis_url_configured"] is True
    assert payload["redis_stream_prefix"] == "shuihuo"
    assert payload["jd_vector_backend"] == "local"
    assert payload["object_storage_endpoint_configured"] is False
    assert payload["object_storage_bucket"] == "shuihuo-killer"
    assert "super-secret" not in response.text


def test_gateway_config_status_redacts_database_password(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:db-secret@localhost:5432/app")
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    get_settings.cache_clear()
    response = TestClient(app).get("/api/config/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["database_url"] == "postgresql://user:***@localhost:5432/app"
    assert "db-secret" not in response.text


def test_gateway_api_key_auth_can_be_enabled(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch, gateway_api_key="gateway-secret")

    assert client.get("/health").status_code == 200
    unauthorized = client.post("/api/jobs", json={"title": "Backend", "jd_text": "Python"})
    assert unauthorized.status_code == 401
    assert unauthorized.headers["x-request-id"]

    created = client.post(
        "/api/jobs",
        headers={"x-api-key": "gateway-secret"},
        json={"title": "Backend", "jd_text": "Python"},
    )
    assert created.status_code == 200
    status = client.get(
        "/api/config/status",
        headers={"authorization": "Bearer gateway-secret"},
    )
    assert status.status_code == 200
    payload = status.json()
    assert payload["gateway_auth_enabled"] is True
    assert "gateway-secret" not in status.text


def test_gateway_websocket_requires_api_key_when_enabled(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch, gateway_api_key="gateway-secret")
    headers = {"x-api-key": "gateway-secret"}
    job = client.post(
        "/api/jobs",
        headers=headers,
        json={"title": "Backend", "jd_text": "Python"},
    ).json()
    candidate = client.post(
        "/api/candidates",
        headers=headers,
        json={"name": "Candidate"},
    ).json()
    interview = client.post(
        "/api/interviews",
        headers=headers,
        json={"job_id": job["id"], "candidate_id": candidate["id"]},
    ).json()

    try:
        with client.websocket_connect(f"/ws/interview/{interview['id']}"):
            raise AssertionError("expected websocket auth failure")
    except WebSocketDisconnect as exc:
        assert exc.code == 1008

    with client.websocket_connect(
        f"/ws/interview/{interview['id']}",
        headers=headers,
    ) as websocket:
        websocket.send_json(
            {
                "type": "text_turn",
                "seq": 1,
                "answer": "我在 FastAPI 项目里负责接口编排、异常重试和 JSON 校验。",
            }
        )
        assert websocket.receive_json()["type"] == "transcript"
        assert websocket.receive_json()["type"] == "probe"
        assert websocket.receive_json()["type"] == "credibility"
        websocket.send_json({"type": "end"})
        assert websocket.receive_json()["type"] == "report"


def test_gateway_job_probe_pattern_search(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    job = client.post(
        "/api/jobs",
        json={"title": "LLM Backend", "jd_text": "Python FastAPI LLM 可靠性"},
    ).json()

    response = client.get(
        f"/api/jobs/{job['id']}/probe-patterns",
        params={"q": "LLM 调用失败降级 FastAPI 异常处理", "limit": 3},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload
    assert payload[0]["job_id"] == job["id"]
    assert any("LLM" in item["pattern"] or "FastAPI" in item["pattern"] for item in payload)


def test_gateway_exposes_internal_aigc_scoring_and_report_contracts(
    tmp_path: Path, monkeypatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    job = client.post(
        "/api/jobs",
        json={"title": "Backend", "jd_text": "Python FastAPI LLM 可靠性"},
    ).json()
    candidate = client.post("/api/candidates", json={"name": "Candidate"}).json()
    interview = client.post(
        "/api/interviews",
        json={"job_id": job["id"], "candidate_id": candidate["id"]},
    ).json()
    interview = client.post(
        f"/api/interviews/{interview['id']}/turns",
        json={
            "question": "讲项目",
            "answer": "我主要负责整体架构设计并推动项目落地最终取得显著提升",
            "answer_start_ms": 0,
            "answer_end_ms": 1000,
        },
    ).json()

    turns = interview["context"]["turns"]
    aigc = client.post("/api/aigc/detect", json={"turns": turns})
    assert aigc.status_code == 200
    aigc_results = aigc.json()
    assert aigc_results[0]["turn_id"] == turns[0]["turn_id"]
    assert aigc_results[0]["flagged"] is True

    score = client.post(
        "/api/scoring/score",
        json={"context": interview["context"], "aigc_results": aigc_results},
    )
    assert score.status_code == 200
    payload = score.json()
    assert payload["session_id"] == interview["id"]
    assert payload["dimensions"]
    assert payload["total_score"] > 0
    assert payload["risk_notes"]

    report = client.post(
        "/api/report/build",
        json={"context": interview["context"], "score": payload, "aigc_results": aigc_results},
    )
    assert report.status_code == 200
    report_payload = report.json()
    assert report_payload["interview_id"] == interview["id"]
    assert report_payload["score"]["total_score"] == payload["total_score"]
    assert report_payload["aigc_results"][0]["turn_id"] == turns[0]["turn_id"]
    assert report_payload["artifact_uris"]["json"].startswith("file://")
    assert report_payload["artifact_uris"]["html"].startswith("file://")
    assert report_payload["artifact_uris"]["pdf"].startswith("file://")
    assert Path(report_payload["json_path"]).read_text(encoding="utf-8")
    assert Path(report_payload["pdf_path"]).read_bytes().startswith(b"%PDF")


def test_gateway_report_build_rejects_mismatched_inputs(
    tmp_path: Path, monkeypatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    job = client.post(
        "/api/jobs",
        json={"title": "Backend", "jd_text": "Python FastAPI LLM 可靠性"},
    ).json()
    candidate = client.post("/api/candidates", json={"name": "Candidate"}).json()
    interview = client.post(
        "/api/interviews",
        json={"job_id": job["id"], "candidate_id": candidate["id"]},
    ).json()
    interview = client.post(
        f"/api/interviews/{interview['id']}/turns",
        json={
            "question": "讲项目",
            "answer": "我写了 FastAPI 编排、模型重试和 JSON 校验，因为线上有格式漂移。",
            "answer_start_ms": 0,
            "answer_end_ms": 1000,
        },
    ).json()
    turns = interview["context"]["turns"]
    aigc_results = client.post("/api/aigc/detect", json={"turns": turns}).json()
    score = client.post(
        "/api/scoring/score",
        json={"context": interview["context"], "aigc_results": aigc_results},
    ).json()

    mismatched_score = {**score, "session_id": "other-session"}
    rejected_score = client.post(
        "/api/report/build",
        json={
            "context": interview["context"],
            "score": mismatched_score,
            "aigc_results": aigc_results,
        },
    )
    assert rejected_score.status_code == 409
    assert "score session_id" in rejected_score.text

    mismatched_dimension = json.loads(json.dumps(score))
    mismatched_dimension["dimensions"][0]["dimension"] = "不存在的维度"
    rejected_dimension = client.post(
        "/api/report/build",
        json={
            "context": interview["context"],
            "score": mismatched_dimension,
            "aigc_results": aigc_results,
        },
    )
    assert rejected_dimension.status_code == 409
    assert "score dimensions must match competency model items" in rejected_dimension.text

    mismatched_weight = json.loads(json.dumps(score))
    mismatched_weight["dimensions"][0]["weight"] = 99
    rejected_weight = client.post(
        "/api/report/build",
        json={
            "context": interview["context"],
            "score": mismatched_weight,
            "aigc_results": aigc_results,
        },
    )
    assert rejected_weight.status_code == 409
    assert "score dimension weight must match competency model" in rejected_weight.text

    mismatched_total = {**score, "total_score": 1}
    rejected_total = client.post(
        "/api/report/build",
        json={
            "context": interview["context"],
            "score": mismatched_total,
            "aigc_results": aigc_results,
        },
    )
    assert rejected_total.status_code == 409
    assert "score total_score must match dimension scores and weights" in rejected_total.text

    mismatched_recommendation = {**score, "recommendation": "no"}
    rejected_recommendation = client.post(
        "/api/report/build",
        json={
            "context": interview["context"],
            "score": mismatched_recommendation,
            "aigc_results": aigc_results,
        },
    )
    assert rejected_recommendation.status_code == 409
    assert "score recommendation must match total_score" in rejected_recommendation.text

    mismatched_evidence = json.loads(json.dumps(score))
    mismatched_evidence["dimensions"][0]["evidence"][0]["turn_id"] = "missing-turn"
    rejected_evidence = client.post(
        "/api/report/build",
        json={
            "context": interview["context"],
            "score": mismatched_evidence,
            "aigc_results": aigc_results,
        },
    )
    assert rejected_evidence.status_code == 409
    assert "score evidence references unknown turn_id" in rejected_evidence.text

    out_of_range_evidence = json.loads(json.dumps(score))
    out_of_range_evidence["dimensions"][0]["evidence"][0]["quote_end_ms"] = 999999
    rejected_evidence_range = client.post(
        "/api/report/build",
        json={
            "context": interview["context"],
            "score": out_of_range_evidence,
            "aigc_results": aigc_results,
        },
    )
    assert rejected_evidence_range.status_code == 409
    assert "score evidence timestamp is outside turn range" in rejected_evidence_range.text

    wrong_excerpt_evidence = json.loads(json.dumps(score))
    wrong_excerpt_evidence["dimensions"][0]["evidence"][0]["excerpt"] = "不存在于回答里的片段"
    rejected_evidence_excerpt = client.post(
        "/api/report/build",
        json={
            "context": interview["context"],
            "score": wrong_excerpt_evidence,
            "aigc_results": aigc_results,
        },
    )
    assert rejected_evidence_excerpt.status_code == 409
    assert "score evidence excerpt is not in turn answer" in rejected_evidence_excerpt.text

    missing_aigc = client.post(
        "/api/report/build",
        json={
            "context": interview["context"],
            "score": score,
            "aigc_results": [],
        },
    )
    assert missing_aigc.status_code == 409
    assert "AIGC results must cover every transcript turn" in missing_aigc.text

    duplicate_aigc = [aigc_results[0], aigc_results[0]]
    rejected_duplicate_aigc = client.post(
        "/api/report/build",
        json={
            "context": interview["context"],
            "score": score,
            "aigc_results": duplicate_aigc,
        },
    )
    assert rejected_duplicate_aigc.status_code == 409
    assert "AIGC results must not contain duplicate turn_id values" in rejected_duplicate_aigc.text

    threshold_bypass_aigc = [
        {
            **aigc_results[0],
            "ai_generated_prob": 0.99,
            "template_similarity": 0.0,
            "flagged": False,
        }
    ]
    rejected_threshold_bypass = client.post(
        "/api/report/build",
        json={
            "context": interview["context"],
            "score": score,
            "aigc_results": threshold_bypass_aigc,
        },
    )
    assert rejected_threshold_bypass.status_code == 409
    assert "AIGC result flagged must be true when thresholds are exceeded" in (
        rejected_threshold_bypass.text
    )

    mismatched_aigc = [{**aigc_results[0], "turn_id": "missing-turn"}]
    rejected_aigc = client.post(
        "/api/report/build",
        json={
            "context": interview["context"],
            "score": score,
            "aigc_results": mismatched_aigc,
        },
    )
    assert rejected_aigc.status_code == 409
    assert "AIGC result references unknown turn_id" in rejected_aigc.text


def test_gateway_end_interview_can_return_queued_task(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OFFLINE_TASK_EXECUTION", "async")
    client = _client(tmp_path, monkeypatch)
    job = client.post("/api/jobs", json={"title": "Backend", "jd_text": "Python"}).json()
    candidate = client.post("/api/candidates", json={"name": "Candidate"}).json()
    interview = client.post(
        "/api/interviews",
        json={"job_id": job["id"], "candidate_id": candidate["id"]},
    ).json()
    client.post(
        f"/api/interviews/{interview['id']}/turns",
        json={"question": "讲项目", "answer": "我写了 FastAPI 编排。"},
    )

    response = client.post(f"/api/interviews/{interview['id']}/end")

    assert response.status_code == 200
    payload = response.json()
    assert payload["interview_id"] == interview["id"]
    assert payload["status"] == "queued"
    assert payload["task_name"] == "interview.offline_scoring"
    assert client.get(f"/api/interviews/{interview['id']}/report").status_code == 404


def test_gateway_rejects_turns_after_report(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    job = client.post("/api/jobs", json={"title": "Backend", "jd_text": "Python"}).json()
    candidate = client.post("/api/candidates", json={"name": "Candidate"}).json()
    interview = client.post(
        "/api/interviews",
        json={"job_id": job["id"], "candidate_id": candidate["id"]},
    ).json()
    client.post(
        f"/api/interviews/{interview['id']}/turns",
        json={"question": "讲项目", "answer": "我写了 FastAPI 编排。"},
    )
    assert client.post(f"/api/interviews/{interview['id']}/end").status_code == 200

    rejected = client.post(
        f"/api/interviews/{interview['id']}/turns",
        json={"question": "补充", "answer": "继续补充"},
    )

    assert rejected.status_code == 409
    assert "cannot add turn" in rejected.text


def test_gateway_websocket_reports_state_errors(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    job = client.post("/api/jobs", json={"title": "Backend", "jd_text": "Python"}).json()
    candidate = client.post("/api/candidates", json={"name": "Candidate"}).json()
    interview = client.post(
        "/api/interviews",
        json={"job_id": job["id"], "candidate_id": candidate["id"]},
    ).json()
    client.post(
        f"/api/interviews/{interview['id']}/turns",
        json={"question": "讲项目", "answer": "我写了 FastAPI 编排。"},
    )
    assert client.post(f"/api/interviews/{interview['id']}/end").status_code == 200

    with client.websocket_connect(f"/ws/interview/{interview['id']}") as websocket:
        error = websocket.receive_json()

    assert error["type"] == "error"
    assert "cannot start interview from status REPORTED" in error["detail"]


def test_gateway_metrics_records_requests(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    traceparent = "00-1234567890abcdef1234567890abcdef-1234567890abcdef-01"
    health = client.get(
        "/health",
        headers={"x-request-id": "trace-123", "traceparent": traceparent},
    )
    assert health.status_code == 200
    assert health.headers["x-request-id"] == "trace-123"
    assert health.headers["traceparent"].startswith("00-1234567890abcdef1234567890abcdef-")
    assert health.headers["traceparent"] != traceparent

    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert 'shuihuo_http_requests_total{method="GET",path="/health",status="200"} 1' in (
        response.text
    )


def test_gateway_metrics_include_offline_task_events(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    job = client.post("/api/jobs", json={"title": "Backend", "jd_text": "Python"}).json()
    candidate = client.post("/api/candidates", json={"name": "Candidate"}).json()
    interview = client.post(
        "/api/interviews",
        json={"job_id": job["id"], "candidate_id": candidate["id"]},
    ).json()
    client.post(
        f"/api/interviews/{interview['id']}/turns",
        json={"question": "讲项目", "answer": "我写了 FastAPI 编排。"},
    )
    assert client.post(f"/api/interviews/{interview['id']}/end").status_code == 200

    metrics = client.get("/metrics").text

    assert 'shuihuo_events_total{topic="task.enqueued"} 1' in metrics
    assert 'shuihuo_events_total{topic="task.completed"} 1' in metrics
    assert 'shuihuo_events_total{topic="interview.finished"} 1' in metrics
    assert 'shuihuo_events_total{topic="interview.reported"} 1' in metrics


def test_gateway_writes_structured_request_log(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    client = _client(tmp_path, monkeypatch)
    caplog.set_level(logging.INFO, logger="shuihuo")

    response = client.get("/health", headers={"x-request-id": "trace-log"})

    assert response.status_code == 200
    payloads = [json.loads(record.message) for record in caplog.records if record.name == "shuihuo"]
    request_log = next(item for item in payloads if item["event"] == "http.request")
    assert request_log["request_id"] == "trace-log"
    assert len(request_log["trace_id"]) == 32
    assert len(request_log["span_id"]) == 16
    assert request_log["method"] == "GET"
    assert request_log["path"] == "/health"
    assert request_log["status_code"] == 200
    assert request_log["duration_seconds"] >= 0


def test_gateway_rate_limit_can_be_enabled(tmp_path: Path, monkeypatch) -> None:
    client = _client(
        tmp_path,
        monkeypatch,
        rate_limit_enabled=True,
        rate_limit_requests_per_minute=2,
    )

    assert client.get("/health").status_code == 200
    assert client.get("/health").status_code == 200
    limited = client.get("/health")

    assert limited.status_code == 429
    assert limited.headers["retry-after"].isdigit()


def test_signal_requires_admin_enablement(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    job = client.post("/api/jobs", json={"title": "Backend", "jd_text": "Python"}).json()
    candidate = client.post("/api/candidates", json={"name": "Candidate"}).json()

    consent = client.post(
        "/api/consents",
        json={"candidate_id": candidate["id"], "consent_type": "behavior_signal", "granted": True},
    )
    assert consent.status_code == 200
    rejected = client.post(
        "/api/interviews",
        json={"job_id": job["id"], "candidate_id": candidate["id"], "signal_enabled": True},
    )
    assert rejected.status_code == 403
    assert "admin enablement" in rejected.text


def test_gateway_rejects_interview_for_missing_candidate(
    tmp_path: Path, monkeypatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    job = client.post("/api/jobs", json={"title": "Backend", "jd_text": "Python"}).json()

    rejected = client.post(
        "/api/interviews",
        json={"job_id": job["id"], "candidate_id": "missing-candidate"},
    )

    assert rejected.status_code == 404
    assert "candidate not found" in rejected.text


def test_gateway_rejects_consent_for_missing_candidate(
    tmp_path: Path, monkeypatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    rejected = client.post(
        "/api/consents",
        json={
            "candidate_id": "missing-candidate",
            "consent_type": "behavior_signal",
            "granted": True,
        },
    )

    assert rejected.status_code == 404
    assert "candidate not found" in rejected.text


def test_signal_requires_candidate_consent(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch, signal_enabled=True)
    job = client.post("/api/jobs", json={"title": "Backend", "jd_text": "Python"}).json()
    candidate = client.post("/api/candidates", json={"name": "Candidate"}).json()

    rejected = client.post(
        "/api/interviews",
        json={"job_id": job["id"], "candidate_id": candidate["id"], "signal_enabled": True},
    )
    assert rejected.status_code == 403
    assert "explicit candidate consent" in rejected.text

    consent = client.post(
        "/api/consents",
        json={"candidate_id": candidate["id"], "consent_type": "behavior_signal", "granted": True},
    )
    assert consent.status_code == 200
    accepted = client.post(
        "/api/interviews",
        json={"job_id": job["id"], "candidate_id": candidate["id"], "signal_enabled": True},
    )
    assert accepted.status_code == 200
    assert accepted.json()["signal_enabled"] is True


def test_signal_consent_can_be_revoked(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch, signal_enabled=True)
    job = client.post("/api/jobs", json={"title": "Backend", "jd_text": "Python"}).json()
    candidate = client.post("/api/candidates", json={"name": "Candidate"}).json()

    granted = client.post(
        "/api/consents",
        json={"candidate_id": candidate["id"], "consent_type": "behavior_signal", "granted": True},
    )
    assert granted.status_code == 200
    accepted = client.post(
        "/api/interviews",
        json={"job_id": job["id"], "candidate_id": candidate["id"], "signal_enabled": True},
    )
    assert accepted.status_code == 200

    revoked = client.post(
        "/api/consents",
        json={"candidate_id": candidate["id"], "consent_type": "behavior_signal", "granted": False},
    )
    assert revoked.status_code == 200
    assert revoked.json()["granted"] is False
    assert revoked.json()["revoked_at"] is not None

    rejected = client.post(
        "/api/interviews",
        json={"job_id": job["id"], "candidate_id": candidate["id"], "signal_enabled": True},
    )
    assert rejected.status_code == 403


def test_gateway_websocket_emits_behavior_signal_with_active_consent(
    tmp_path: Path, monkeypatch
) -> None:
    client = _client(tmp_path, monkeypatch, signal_enabled=True)
    job = client.post("/api/jobs", json={"title": "Backend", "jd_text": "Python"}).json()
    candidate = client.post("/api/candidates", json={"name": "Candidate"}).json()
    client.post(
        "/api/consents",
        json={"candidate_id": candidate["id"], "consent_type": "behavior_signal", "granted": True},
    )
    interview = client.post(
        "/api/interviews",
        json={"job_id": job["id"], "candidate_id": candidate["id"], "signal_enabled": True},
    ).json()

    with client.websocket_connect(f"/ws/interview/{interview['id']}") as websocket:
        websocket.send_json(
            {
                "type": "text_turn",
                "seq": 1,
                "answer": "嗯这个项目里我主要负责 FastAPI 编排、重试和 JSON 校验。",
            }
        )
        assert websocket.receive_json()["type"] == "transcript"
        assert websocket.receive_json()["type"] == "probe"
        assert websocket.receive_json()["type"] == "credibility"
        signal = websocket.receive_json()

    assert signal["type"] == "signal"
    assert set(signal["payload"]) == {"turn_id", "fluency", "hesitation", "evasiveness_hint"}
    assert signal["payload"]["hesitation"] > 0


def test_gateway_websocket_suppresses_behavior_signal_after_consent_revocation(
    tmp_path: Path, monkeypatch
) -> None:
    client = _client(tmp_path, monkeypatch, signal_enabled=True)
    job = client.post("/api/jobs", json={"title": "Backend", "jd_text": "Python"}).json()
    candidate = client.post("/api/candidates", json={"name": "Candidate"}).json()
    client.post(
        "/api/consents",
        json={"candidate_id": candidate["id"], "consent_type": "behavior_signal", "granted": True},
    )
    interview = client.post(
        "/api/interviews",
        json={"job_id": job["id"], "candidate_id": candidate["id"], "signal_enabled": True},
    ).json()
    client.post(
        "/api/consents",
        json={"candidate_id": candidate["id"], "consent_type": "behavior_signal", "granted": False},
    )

    with client.websocket_connect(f"/ws/interview/{interview['id']}") as websocket:
        websocket.send_json(
            {
                "type": "text_turn",
                "seq": 1,
                "answer": "嗯这个项目里我主要负责 FastAPI 编排、重试和 JSON 校验。",
            }
        )
        assert websocket.receive_json()["type"] == "transcript"
        assert websocket.receive_json()["type"] == "probe"
        assert websocket.receive_json()["type"] == "credibility"
        websocket.send_json({"type": "end"})
        report = websocket.receive_json()

    assert report["type"] == "report"
    assert "signal" not in json.dumps(report["payload"], ensure_ascii=False)


def test_gateway_websocket_probe_flow(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    job = client.post("/api/jobs", json={"title": "Backend", "jd_text": "Python"}).json()
    candidate = client.post("/api/candidates", json={"name": "Candidate"}).json()
    interview = client.post(
        "/api/interviews",
        json={"job_id": job["id"], "candidate_id": candidate["id"]},
    ).json()

    text = "我主要负责优化，做了很多事情，效果比较好。"
    audio = base64.b64encode(text.encode("utf-8")).decode("ascii")
    with client.websocket_connect(f"/ws/interview/{interview['id']}") as websocket:
        websocket.send_json(
            {
                "type": "audio_chunk",
                "session_id": interview["id"],
                "seq": 1,
                "audio": audio,
                "audio_format": "pcm",
                "sample_rate_hz": 16000,
                "channels": 1,
            }
        )
        transcript = websocket.receive_json()
        probe = websocket.receive_json()
        credibility = websocket.receive_json()
        assert transcript["type"] == "transcript"
        assert probe["type"] == "probe"
        assert probe["payload"]["credibility"]["level"] in {"vague", "suspicious"}
        assert credibility["type"] == "credibility"
        assert credibility["payload"]["level"] == probe["payload"]["credibility"]["level"]


def test_gateway_websocket_text_turn_probe_flow(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    job = client.post("/api/jobs", json={"title": "Backend", "jd_text": "Python"}).json()
    candidate = client.post("/api/candidates", json={"name": "Candidate"}).json()
    interview = client.post(
        "/api/interviews",
        json={"job_id": job["id"], "candidate_id": candidate["id"]},
    ).json()

    with client.websocket_connect(f"/ws/interview/{interview['id']}") as websocket:
        websocket.send_json(
            {
                "type": "text_turn",
                "seq": 1,
                "answer": "我主要负责优化，做了很多事情，效果比较好。",
            }
        )
        transcript = websocket.receive_json()
        probe = websocket.receive_json()
        credibility = websocket.receive_json()
        assert transcript["type"] == "transcript"
        assert transcript["payload"]["speaker"] == "candidate"
        assert probe["type"] == "probe"
        assert probe["payload"]["suggestions"]
        assert credibility["type"] == "credibility"


def test_gateway_websocket_manual_probe_bypasses_auto_gates(
    tmp_path: Path, monkeypatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    job = client.post("/api/jobs", json={"title": "Backend", "jd_text": "Python"}).json()
    candidate = client.post("/api/candidates", json={"name": "Candidate"}).json()
    interview = client.post(
        "/api/interviews",
        json={"job_id": job["id"], "candidate_id": candidate["id"]},
    ).json()

    with client.websocket_connect(f"/ws/interview/{interview['id']}") as websocket:
        websocket.send_json({"type": "manual_probe", "answer": "好"})
        probe = websocket.receive_json()
        credibility = websocket.receive_json()
        assert probe["type"] == "probe"
        assert probe["payload"]["suggestions"]
        assert credibility["type"] == "credibility"


def test_gateway_websocket_ignores_non_final_and_interviewer_segments(
    tmp_path: Path, monkeypatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    job = client.post("/api/jobs", json={"title": "Backend", "jd_text": "Python"}).json()
    candidate = client.post("/api/candidates", json={"name": "Candidate"}).json()
    interview = client.post(
        "/api/interviews",
        json={"job_id": job["id"], "candidate_id": candidate["id"]},
    ).json()

    text = "我主要负责优化，做了很多事情，效果比较好。"
    audio = base64.b64encode(text.encode("utf-8")).decode("ascii")
    with client.websocket_connect(f"/ws/interview/{interview['id']}") as websocket:
        websocket.send_json(
            {
                "type": "audio_chunk",
                "session_id": interview["id"],
                "seq": 1,
                "audio": audio,
                "speaker": "candidate",
                "is_final": "partial",
                "start_ms": 100,
                "end_ms": 600,
                "confidence": 0.6,
            }
        )
        transcript = websocket.receive_json()
        assert transcript["type"] == "transcript"
        assert transcript["payload"]["speaker"] == "candidate"
        assert transcript["payload"]["is_final"] is False
        assert transcript["payload"]["start_ms"] == 100
        assert transcript["payload"]["end_ms"] == 600
        assert transcript["payload"]["confidence"] == 0.6

        websocket.send_json(
            {
                "type": "audio_chunk",
                "session_id": interview["id"],
                "seq": 3,
                "audio": audio,
                "speaker": "interviewer",
                "is_final": True,
            }
        )
        interviewer_transcript = websocket.receive_json()
        assert interviewer_transcript["type"] == "transcript"
        assert interviewer_transcript["payload"]["speaker"] == "interviewer"

        websocket.send_json({"type": "end"})
        error = websocket.receive_json()
        assert error["type"] == "error"
        assert "cannot finish interview without candidate turns" in error["detail"]


def test_gateway_websocket_rejects_invalid_audio_base64(
    tmp_path: Path, monkeypatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    job = client.post("/api/jobs", json={"title": "Backend", "jd_text": "Python"}).json()
    candidate = client.post("/api/candidates", json={"name": "Candidate"}).json()
    interview = client.post(
        "/api/interviews",
        json={"job_id": job["id"], "candidate_id": candidate["id"]},
    ).json()

    with client.websocket_connect(f"/ws/interview/{interview['id']}") as websocket:
        websocket.send_json(
            {
                "type": "audio_chunk",
                "session_id": interview["id"],
                "seq": 1,
                "audio": "not-base64!",
                "speaker": "candidate",
            }
        )
        warning = websocket.receive_json()
        websocket.send_json({"type": "end"})
        error = websocket.receive_json()

    assert warning["type"] == "asr_warning"
    assert warning["payload"] == {"reason": "invalid_audio_base64", "seq": 1}
    assert error["type"] == "error"
    assert "cannot finish interview without candidate turns" in error["detail"]


def test_gateway_websocket_rejects_unsupported_audio_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    job = client.post("/api/jobs", json={"title": "Backend", "jd_text": "Python"}).json()
    candidate = client.post("/api/candidates", json={"name": "Candidate"}).json()
    interview = client.post(
        "/api/interviews",
        json={"job_id": job["id"], "candidate_id": candidate["id"]},
    ).json()
    audio = base64.b64encode("我负责 FastAPI 编排和重试。".encode("utf-8")).decode("ascii")

    with client.websocket_connect(f"/ws/interview/{interview['id']}") as websocket:
        websocket.send_json(
            {
                "type": "audio_chunk",
                "session_id": interview["id"],
                "seq": 1,
                "audio": audio,
                "audio_format": "mp3",
            }
        )
        bad_format = websocket.receive_json()
        websocket.send_json(
            {
                "type": "audio_chunk",
                "session_id": interview["id"],
                "seq": 2,
                "audio": audio,
                "sample_rate_hz": 48000,
            }
        )
        bad_rate = websocket.receive_json()
        websocket.send_json(
            {
                "type": "audio_chunk",
                "session_id": interview["id"],
                "seq": 3,
                "audio": audio,
                "channels": 2,
            }
        )
        bad_channels = websocket.receive_json()
        websocket.send_json({"type": "end"})
        error = websocket.receive_json()

    assert bad_format["type"] == "asr_warning"
    assert bad_format["payload"] == {"reason": "unsupported_audio_format", "seq": 1}
    assert bad_rate["type"] == "asr_warning"
    assert bad_rate["payload"] == {"reason": "unsupported_sample_rate", "seq": 2}
    assert bad_channels["type"] == "asr_warning"
    assert bad_channels["payload"] == {"reason": "unsupported_channel_count", "seq": 3}
    assert error["type"] == "error"
    assert "cannot finish interview without candidate turns" in error["detail"]


def test_gateway_websocket_rejects_mismatched_audio_session(
    tmp_path: Path, monkeypatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    job = client.post("/api/jobs", json={"title": "Backend", "jd_text": "Python"}).json()
    candidate = client.post("/api/candidates", json={"name": "Candidate"}).json()
    interview = client.post(
        "/api/interviews",
        json={"job_id": job["id"], "candidate_id": candidate["id"]},
    ).json()
    audio = base64.b64encode("我负责 FastAPI 编排和重试。".encode("utf-8")).decode("ascii")

    with client.websocket_connect(f"/ws/interview/{interview['id']}") as websocket:
        websocket.send_json(
            {
                "type": "audio_chunk",
                "session_id": "other-session",
                "seq": 7,
                "audio": audio,
                "speaker": "candidate",
            }
        )
        warning = websocket.receive_json()
        websocket.send_json({"type": "end"})
        error = websocket.receive_json()

    assert warning["type"] == "asr_warning"
    assert warning["payload"] == {"reason": "session_id_mismatch", "seq": 7}
    assert error["type"] == "error"
    assert "cannot finish interview without candidate turns" in error["detail"]


def test_gateway_websocket_maps_audio_channel_to_speaker(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    job = client.post("/api/jobs", json={"title": "Backend", "jd_text": "Python"}).json()
    candidate = client.post("/api/candidates", json={"name": "Candidate"}).json()
    interview = client.post(
        "/api/interviews",
        json={"job_id": job["id"], "candidate_id": candidate["id"]},
    ).json()

    text = "我主要负责优化，做了很多事情，效果比较好。"
    audio = base64.b64encode(text.encode("utf-8")).decode("ascii")
    with client.websocket_connect(f"/ws/interview/{interview['id']}") as websocket:
        websocket.send_json(
            {
                "type": "audio_chunk",
                "session_id": interview["id"],
                "seq": 2,
                "audio": audio,
                "channel": "right",
                "is_final": True,
            }
        )
        transcript = websocket.receive_json()
        probe = websocket.receive_json()
        credibility = websocket.receive_json()
        assert transcript["type"] == "transcript"
        assert transcript["payload"]["speaker"] == "candidate"
        assert probe["type"] == "probe"
        assert credibility["type"] == "credibility"

        interviewer_text = "请介绍一下项目背景。"
        interviewer_audio = base64.b64encode(interviewer_text.encode("utf-8")).decode("ascii")
        websocket.send_json(
            {
                "type": "audio_chunk",
                "session_id": interview["id"],
                "seq": 3,
                "audio": interviewer_audio,
                "channel": "left",
                "is_final": True,
            }
        )
        interviewer_transcript = websocket.receive_json()
        assert interviewer_transcript["type"] == "transcript"
        assert interviewer_transcript["payload"]["speaker"] == "interviewer"

        websocket.send_json({"type": "end"})
        report = websocket.receive_json()
        assert report["type"] == "report"


def test_gateway_websocket_deduplicates_final_audio_segments(
    tmp_path: Path, monkeypatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    job = client.post("/api/jobs", json={"title": "Backend", "jd_text": "Python"}).json()
    candidate = client.post("/api/candidates", json={"name": "Candidate"}).json()
    interview = client.post(
        "/api/interviews",
        json={"job_id": job["id"], "candidate_id": candidate["id"]},
    ).json()

    text = "我主要负责优化，做了很多事情，效果比较好。"
    audio = base64.b64encode(text.encode("utf-8")).decode("ascii")
    event = {
        "type": "audio_chunk",
        "session_id": interview["id"],
        "seq": 1,
        "audio": audio,
        "speaker": "candidate",
        "is_final": True,
    }
    with client.websocket_connect(f"/ws/interview/{interview['id']}") as websocket:
        websocket.send_json(event)
        assert websocket.receive_json()["type"] == "transcript"
        assert websocket.receive_json()["type"] == "probe"
        assert websocket.receive_json()["type"] == "credibility"

        websocket.send_json(event)
        warning = websocket.receive_json()
        assert warning["type"] == "asr_warning"
        assert warning["payload"]["reason"] == "duplicate_final_segment"

        websocket.send_json({"type": "end"})
        assert websocket.receive_json()["type"] == "report"


def test_gateway_one_shot_offline_evaluate(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    response = client.post(
        "/api/offline/evaluate",
        json={
            "job_title": "AI 面试系统后端",
            "jd_text": "Python FastAPI LLM 报告生成",
            "candidate_name": "Candidate",
            "resume_text": "做过 AI 应用",
            "turns": [
                {
                    "question": "介绍最核心的项目",
                    "answer": "我主要负责整体架构设计并推动项目落地最终取得显著提升",
                    "answer_start_ms": 0,
                    "answer_end_ms": 1000,
                },
                {
                    "question": "具体你写了哪部分",
                    "answer": "我写了 FastAPI 编排、模型重试和 JSON 校验，因为线上有格式漂移。",
                    "answer_start_ms": 1200,
                    "answer_end_ms": 3000,
                },
            ],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["report"]["score"]["total_score"] > 0
    assert payload["interview"]["status"] == "REPORTED"
    report_id = payload["report"]["interview_id"]
    assert client.get(f"/api/interviews/{report_id}/report").status_code == 200
