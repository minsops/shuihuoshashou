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
    pdf = client.get(f"/api/interviews/{interview['id']}/report.pdf")
    assert pdf.status_code == 200
    assert pdf.headers["content-type"] == "application/pdf"


def test_gateway_serves_demo_ui(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    response = client.get("/")
    assert response.status_code == 200
    assert "水货杀手" in response.text
    assert "/api/offline/evaluate" in response.text


def test_gateway_config_status_hides_secrets(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "super-secret")
    client = _client(tmp_path, monkeypatch)
    response = client.get("/api/config/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["llm_api_key_configured"] is True
    assert payload["llm_max_retries"] == 1
    assert payload["gateway_auth_enabled"] is False
    assert payload["asr_provider"] == "stub"
    assert payload["asr_base_url_configured"] is False
    assert payload["asr_api_key_configured"] is False
    assert payload["asr_channel_diarization_configured"] is True
    assert payload["probe_min_answer_chars"] == 20
    assert payload["probe_min_interval_ms"] == 1000
    assert payload["probe_require_topic_match"] is True
    assert payload["probe_topic_keywords_configured"] is True
    assert payload["speaker_diarization_provider"] == "local"
    assert payload["speaker_diarization_base_url_configured"] is False
    assert payload["speaker_diarization_api_key_configured"] is False
    assert payload["aigc_detector_provider"] == "local"
    assert payload["aigc_detector_base_url_configured"] is False
    assert payload["aigc_detector_api_key_configured"] is False
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
    assert payload["redis_url_configured"] is True
    assert payload["redis_stream_prefix"] == "shuihuo"
    assert payload["jd_vector_backend"] == "local"
    assert payload["object_storage_endpoint_configured"] is False
    assert payload["object_storage_bucket"] == "shuihuo-killer"
    assert "super-secret" not in response.text


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
            {"type": "audio_chunk", "session_id": interview["id"], "seq": 1, "audio": audio}
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
                "is_final": False,
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
        report = websocket.receive_json()
        assert report["type"] == "report"


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
