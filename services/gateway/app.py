from __future__ import annotations

import base64
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

import httpx
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from starlette.datastructures import Headers
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response

from libs.common.config import get_settings
from libs.common.database import init_db
from libs.common.observability import (
    configure_opentelemetry,
    get_rate_limiter,
    log_event,
    metrics_registry,
    request_id_from_header,
    trace_context_from_header,
)
from libs.common.runtime import RuntimeStatus, get_runtime_status
from libs.common.storage import get_artifact_store
from libs.schemas import (
    AIGCDetectRequest,
    CandidateCreate,
    ConsentCreate,
    InterviewCreate,
    JobCreate,
    OfflineInterviewInput,
    OfflineInterviewResult,
    OfflineTaskAccepted,
    ProbeRequest,
    QATurn,
    ReportBuildRequest,
    ScoringRequest,
    TranscriptSegment,
)
from services.aigc_detect_service.service import detect_interview
from services.asr_service.service import (
    ASREngine,
    asr_session_manager,
    configure_asr_runtime,
    get_asr_engine,
)
from services.interview_orchestrator.service import (
    add_turn,
    create_candidate,
    create_consent,
    create_interview,
    end_interview,
    get_interview,
    get_report,
    has_active_consent,
    should_probe,
    start_interview,
)
from services.jd_kb_service.service import create_job, get_job, retrieve_job_probe_patterns
from services.probe_service.service import generate_probe
from services.report_service.service import build_report
from services.scoring_service.service import score_interview
from services.signal_service.service import extract_behavior_signal

VALID_SPEAKERS = {"interviewer", "candidate", "unknown"}
SUPPORTED_AUDIO_FORMATS = {
    "pcm",
    "pcm16",
    "pcm-16",
    "linear16",
    "linear-16",
    "s16le",
    "opus",
}
EXPECTED_AUDIO_SAMPLE_RATE_HZ = 16000
EXPECTED_AUDIO_CHANNELS = 1
FALSE_FINALITY_VALUES = {
    "",
    "0",
    "false",
    "no",
    "off",
    "partial",
    "interim",
    "intermediate",
    "non_final",
    "non-final",
    "not_final",
    "not-final",
    "provisional",
}
TRUE_FINALITY_VALUES = {"1", "true", "yes", "on", "final", "finalized", "complete", "completed"}


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    configure_opentelemetry(fastapi_app, get_settings())
    configure_asr_runtime()
    init_db()
    yield


app = FastAPI(title="Shuihuo Killer", version="0.1.0", lifespan=lifespan)
WEB_INDEX = Path(__file__).resolve().parents[2] / "web" / "index.html"


def _client_key(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",", maxsplit=1)[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def _route_path(request: Request) -> str:
    route = request.scope.get("route")
    return getattr(route, "path", request.url.path)


def _extract_gateway_api_key(headers: Headers, query_key: str | None = None) -> str:
    header_key = headers.get("x-api-key", "")
    if header_key:
        return header_key
    authorization = headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return query_key or ""


def _gateway_authorized(headers: Headers, expected_key: str, query_key: str | None = None) -> bool:
    if not expected_key:
        return True
    return _extract_gateway_api_key(headers, query_key) == expected_key


@app.middleware("http")
async def observe_and_rate_limit(request: Request, call_next):
    settings = get_settings()
    start = perf_counter()
    status_code = 500
    request_id = request_id_from_header(request.headers.get("x-request-id"))
    trace_context = trace_context_from_header(request.headers.get("traceparent"))
    path = request.url.path
    if request.url.path.startswith("/api/") and not _gateway_authorized(
        request.headers,
        settings.gateway_api_key,
        request.query_params.get("api_key"),
    ):
        status_code = 401
        duration = perf_counter() - start
        response = PlainTextResponse(
            "unauthorized",
            status_code=status_code,
            headers={"X-Request-ID": request_id, "traceparent": trace_context.traceparent},
        )
        metrics_registry.record_request(
            request.method, path, status_code, duration
        )
        log_event(
            "http.request",
            request_id=request_id,
            trace_id=trace_context.trace_id,
            span_id=trace_context.span_id,
            method=request.method,
            path=path,
            status_code=status_code,
            duration_seconds=round(duration, 6),
        )
        return response

    if settings.rate_limit_enabled:
        decision = get_rate_limiter(settings).check(_client_key(request))
        if not decision.allowed:
            status_code = 429
            duration = perf_counter() - start
            response = PlainTextResponse(
                "rate limit exceeded",
                status_code=status_code,
                headers={
                    "Retry-After": str(decision.retry_after_seconds),
                    "X-Request-ID": request_id,
                    "traceparent": trace_context.traceparent,
                },
            )
            metrics_registry.record_request(
                request.method, path, status_code, duration
            )
            log_event(
                "http.request",
                request_id=request_id,
                trace_id=trace_context.trace_id,
                span_id=trace_context.span_id,
                method=request.method,
                path=path,
                status_code=status_code,
                duration_seconds=round(duration, 6),
            )
            return response

    try:
        response: Response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        response.headers["traceparent"] = trace_context.traceparent
        return response
    finally:
        duration = perf_counter() - start
        route_path = _route_path(request)
        metrics_registry.record_request(
            request.method, route_path, status_code, duration
        )
        log_event(
            "http.request",
            request_id=request_id,
            trace_id=trace_context.trace_id,
            span_id=trace_context.span_id,
            method=request.method,
            path=route_path,
            status_code=status_code,
            duration_seconds=round(duration, 6),
        )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    return metrics_registry.render_prometheus()


@app.get("/api/config/status", response_model=RuntimeStatus)
def config_status() -> RuntimeStatus:
    return get_runtime_status()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return WEB_INDEX.read_text(encoding="utf-8")


async def _send_probe_for_segment(websocket: WebSocket, interview_id: str, record, segment):
    turn = QATurn(
        question="实时输入片段",
        answer=segment.text,
        answer_start_ms=segment.start_ms,
        answer_end_ms=segment.end_ms,
    )
    record = add_turn(interview_id, turn)
    probe = await generate_probe(
        ProbeRequest(
            job_id=record.job_id,
            competency_model=record.context.competency_model,
            recent_turns=record.context.turns[-5:],
            latest_answer=segment.text,
        )
    )
    await websocket.send_json({"type": "probe", "payload": probe.model_dump()})
    await websocket.send_json({"type": "credibility", "payload": probe.credibility.model_dump()})
    signal = (
        extract_behavior_signal(turn)
        if record.signal_enabled and has_active_consent(record.candidate_id, "behavior_signal")
        else None
    )
    if signal is not None:
        await websocket.send_json({"type": "signal", "payload": signal.model_dump()})
    return record


def _manual_probe_segment(interview_id: str, event: dict) -> TranscriptSegment:
    text = str(event.get("answer") or event.get("latest_answer") or "").strip()
    start_ms = _manual_probe_int(event, "start_ms", 0)
    end_ms = _manual_probe_int(event, "end_ms", start_ms)
    if end_ms < start_ms:
        raise ValueError("manual_probe end_ms must be greater than or equal to start_ms")
    return TranscriptSegment(
        session_id=interview_id,
        speaker="candidate",
        text=text,
        start_ms=start_ms,
        end_ms=end_ms,
        is_final=True,
        confidence=_manual_probe_confidence(event),
    )


def _manual_probe_int(event: dict, key: str, default: int) -> int:
    value = event.get(key)
    if value is None or value == "":
        return default
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        raise ValueError(f"manual_probe {key} must be a non-negative integer") from None
    if parsed < 0:
        raise ValueError(f"manual_probe {key} must be a non-negative integer")
    return parsed


def _manual_probe_confidence(event: dict) -> float:
    value = event.get("confidence")
    if value is None or value == "":
        return 1.0
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        raise ValueError("manual_probe confidence must be a number") from None
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("manual_probe confidence must be between 0 and 1")
    return confidence


async def _send_asr_warning(websocket: WebSocket, reason: str, seq: int) -> None:
    await websocket.send_json({"type": "asr_warning", "payload": {"reason": reason, "seq": seq}})


async def _transcribe_or_warn(
    websocket: WebSocket,
    engine: ASREngine,
    *,
    seq: int,
    session_id: str,
    audio_b64: str,
    speaker: str | None = None,
    start_ms: object = None,
    end_ms: object = None,
    is_final: bool = True,
    confidence: object = None,
) -> TranscriptSegment | None:
    try:
        return await engine.transcribe_chunk(
            session_id=session_id,
            seq=seq,
            audio_b64=audio_b64,
            speaker=speaker,
            start_ms=start_ms,
            end_ms=end_ms,
            is_final=is_final,
            confidence=confidence,
        )
    except (httpx.HTTPError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        log_event(
            "asr.transcription.failed",
            seq=seq,
            session_id=session_id,
            error_type=type(exc).__name__,
        )
        await _send_asr_warning(websocket, "asr_transcription_failed", seq)
        return None


def _event_speaker(event: dict) -> str | None:
    speaker = event.get("speaker")
    if speaker in VALID_SPEAKERS:
        return str(speaker)
    channel = _event_channel(event)
    if channel is None:
        return None
    settings = get_settings()
    if channel in _channel_aliases(settings.asr_interviewer_channels):
        return "interviewer"
    if channel in _channel_aliases(settings.asr_candidate_channels):
        return "candidate"
    return None


def _event_channel(event: dict) -> str | None:
    for key in ("channel", "audio_channel", "track"):
        if key in event and event[key] is not None:
            return str(event[key]).strip().lower()
    return None


def _event_session_mismatch(interview_id: str, event: dict) -> bool:
    session_id = str(event.get("session_id", "")).strip()
    return bool(session_id and session_id != interview_id)


def _channel_aliases(raw: str) -> set[str]:
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


def _event_bool(event: dict, key: str, default: bool) -> bool:
    value = event.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in FALSE_FINALITY_VALUES:
            return False
        if lowered in TRUE_FINALITY_VALUES:
            return True
    return bool(value)


def _valid_audio_b64(value: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        return bool(base64.b64decode(value, validate=True))
    except Exception:
        return False


def _event_seq(event: dict, default: int) -> int | None:
    value = event.get("seq", default)
    try:
        seq = int(value)
    except (TypeError, ValueError):
        return None
    return seq if seq >= 0 else None


def _audio_contract_warning(event: dict) -> str | None:
    audio_format = _event_string(event, "format", "audio_format", "codec")
    if audio_format is not None and audio_format not in SUPPORTED_AUDIO_FORMATS:
        return "unsupported_audio_format"
    sample_rate_hz = _event_int(event, "sample_rate_hz", "sample_rate", "rate")
    if sample_rate_hz is not None and sample_rate_hz != EXPECTED_AUDIO_SAMPLE_RATE_HZ:
        return "unsupported_sample_rate"
    channels = _event_int(event, "channels", "channel_count")
    if channels is not None and channels != EXPECTED_AUDIO_CHANNELS:
        return "unsupported_channel_count"
    return None


def _event_string(event: dict, *keys: str) -> str | None:
    for key in keys:
        value = event.get(key)
        if value is None:
            continue
        text = str(value).strip().lower().replace("_", "-")
        if text:
            return text
    return None


def _event_int(event: dict, *keys: str) -> int | None:
    for key in keys:
        value = event.get(key)
        if value is None or value == "":
            continue
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return -1
    return None


@app.post("/api/jobs")
def api_create_job(payload: JobCreate):
    return create_job(payload)


@app.get("/api/jobs/{job_id}")
def api_get_job(job_id: str):
    try:
        return get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}/probe-patterns")
def api_job_probe_patterns(job_id: str, q: str = "", limit: int = 5):
    try:
        return retrieve_job_probe_patterns(job_id, q, limit=max(1, min(limit, 20)))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/candidates")
def api_create_candidate(payload: CandidateCreate):
    return create_candidate(payload)


@app.post("/api/consents")
def api_create_consent(payload: ConsentCreate):
    try:
        return create_consent(payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/interviews")
def api_create_interview(payload: InterviewCreate):
    try:
        return create_interview(payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@app.post("/api/interviews/{interview_id}/start")
def api_start_interview(interview_id: str):
    try:
        return start_interview(interview_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/interviews/{interview_id}/turns")
def api_add_turn(interview_id: str, turn: QATurn):
    try:
        return add_turn(interview_id, turn)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/probe")
async def api_probe(payload: ProbeRequest):
    try:
        return await generate_probe(payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/aigc/detect")
def api_aigc_detect(payload: AIGCDetectRequest):
    return detect_interview(payload.turns)


@app.post("/api/scoring/score")
def api_scoring_score(payload: ScoringRequest):
    try:
        return score_interview(payload.context, payload.aigc_results)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/report/build")
def api_report_build(payload: ReportBuildRequest):
    try:
        report, _ = build_report(payload.context, payload.score, payload.aigc_results)
        return report
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/offline/evaluate", response_model=OfflineInterviewResult)
def api_offline_evaluate(payload: OfflineInterviewInput):
    job = create_job(JobCreate(title=payload.job_title, jd_text=payload.jd_text))
    candidate = create_candidate(
        CandidateCreate(name=payload.candidate_name, resume_text=payload.resume_text)
    )
    interview = create_interview(InterviewCreate(job_id=job.id, candidate_id=candidate.id))
    for turn in payload.turns:
        add_turn(interview.id, turn)
    start_interview(interview.id)
    report = end_interview(interview.id, execute_inline=True)
    interview = get_interview(interview.id)
    return OfflineInterviewResult(
        job=job,
        candidate=candidate,
        interview=interview,
        report=report,
    )


@app.post("/api/interviews/{interview_id}/end")
def api_end_interview(interview_id: str):
    try:
        return end_interview(interview_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/interviews/{interview_id}/report")
def api_report(interview_id: str):
    try:
        report, _ = get_report(interview_id)
        return report
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/interviews/{interview_id}/report.html")
def api_report_html(interview_id: str):
    try:
        _, html = get_report(interview_id)
        return HTMLResponse(html)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/interviews/{interview_id}/report.json")
def api_report_json(interview_id: str):
    try:
        report, _ = get_report(interview_id)
        json_path = report.get("json_path")
        if json_path and Path(json_path).exists():
            return FileResponse(
                json_path,
                media_type="application/json",
                filename=f"{interview_id}.report.json",
            )
        return JSONResponse(
            report,
            headers={"content-disposition": f'attachment; filename="{interview_id}.report.json"'},
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/interviews/{interview_id}/report.pdf")
def api_report_pdf(interview_id: str):
    try:
        report, _ = get_report(interview_id)
        pdf_path = report.get("pdf_path")
        if pdf_path and Path(pdf_path).exists():
            return FileResponse(pdf_path, media_type="application/pdf", filename=f"{interview_id}.pdf")
        pdf_uri = report.get("artifact_uris", {}).get("pdf")
        if pdf_uri:
            artifact = get_artifact_store().get_file(pdf_uri)
            return Response(
                content=artifact.content,
                media_type=artifact.content_type or "application/pdf",
                headers={"content-disposition": f'attachment; filename="{interview_id}.pdf"'},
            )
        raise KeyError(f"report pdf not found: {interview_id}")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (OSError, ValueError, httpx.HTTPError) as exc:
        raise HTTPException(status_code=404, detail=f"report pdf not found: {interview_id}") from exc


@app.get("/api/interviews/{interview_id}/report.transcript.json")
def api_report_transcript(interview_id: str):
    try:
        report, _ = get_report(interview_id)
        transcript_path = report.get("transcript_path")
        if transcript_path and Path(transcript_path).exists():
            return FileResponse(
                transcript_path,
                media_type="application/json",
                filename=f"{interview_id}.transcript.json",
            )
        transcript = report.get("transcript")
        if transcript is None:
            raise KeyError(f"report transcript not found: {interview_id}")
        return JSONResponse(
            transcript,
            headers={
                "content-disposition": f'attachment; filename="{interview_id}.transcript.json"'
            },
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.websocket("/ws/interview/{interview_id}")
async def ws_interview(websocket: WebSocket, interview_id: str):
    settings = get_settings()
    if not _gateway_authorized(
        websocket.headers,
        settings.gateway_api_key,
        websocket.query_params.get("api_key"),
    ):
        await websocket.close(code=1008)
        return
    await websocket.accept()
    engine = get_asr_engine()
    try:
        record = start_interview(interview_id)
        while True:
            event = await websocket.receive_json()
            if event.get("type") == "audio_chunk":
                seq = _event_seq(event, 0)
                if seq is None:
                    await _send_asr_warning(websocket, "invalid_seq", 0)
                    continue
                if _event_session_mismatch(interview_id, event):
                    await _send_asr_warning(websocket, "session_id_mismatch", seq)
                    continue
                audio_b64 = event.get("audio", "")
                if not _valid_audio_b64(audio_b64):
                    await _send_asr_warning(websocket, "invalid_audio_base64", seq)
                    continue
                contract_warning = _audio_contract_warning(event)
                if contract_warning is not None:
                    await _send_asr_warning(websocket, contract_warning, seq)
                    continue
                segment = await _transcribe_or_warn(
                    websocket,
                    engine,
                    session_id=interview_id,
                    seq=seq,
                    audio_b64=audio_b64,
                    speaker=_event_speaker(event),
                    start_ms=event.get("start_ms"),
                    end_ms=event.get("end_ms"),
                    is_final=_event_bool(event, "is_final", True),
                    confidence=event.get("confidence"),
                )
                if segment is None:
                    continue
                decision = asr_session_manager.accept_segment(
                    seq,
                    segment,
                    audio_b64=audio_b64,
                )
                if not decision.accepted or decision.segment is None:
                    await _send_asr_warning(websocket, decision.reason, seq)
                    continue
                segment = decision.segment
                await websocket.send_json({"type": "transcript", "payload": segment.model_dump()})
                if should_probe(segment, record):
                    record = await _send_probe_for_segment(websocket, interview_id, record, segment)
            elif event.get("type") == "text_turn":
                seq = _event_seq(event, 1)
                if seq is None:
                    await websocket.send_json({"type": "error", "detail": "invalid seq"})
                    continue
                text = str(event.get("answer", "")).strip()
                if not text:
                    await websocket.send_json(
                        {"type": "error", "detail": "text_turn requires answer"}
                    )
                    continue
                encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
                segment = await _transcribe_or_warn(
                    websocket,
                    engine,
                    session_id=interview_id,
                    seq=seq,
                    audio_b64=encoded,
                    speaker=_event_speaker(event) or "candidate",
                    start_ms=event.get("start_ms"),
                    end_ms=event.get("end_ms"),
                    is_final=_event_bool(event, "is_final", True),
                    confidence=event.get("confidence"),
                )
                if segment is None:
                    continue
                decision = asr_session_manager.accept_segment(seq, segment, audio_b64=encoded)
                if not decision.accepted or decision.segment is None:
                    await _send_asr_warning(websocket, decision.reason, seq)
                    continue
                segment = decision.segment
                await websocket.send_json({"type": "transcript", "payload": segment.model_dump()})
                if should_probe(segment, record):
                    record = await _send_probe_for_segment(websocket, interview_id, record, segment)
            elif event.get("type") == "manual_probe":
                if not str(event.get("answer") or event.get("latest_answer") or "").strip():
                    await websocket.send_json(
                        {"type": "error", "detail": "manual_probe requires answer"}
                    )
                    continue
                try:
                    segment = _manual_probe_segment(interview_id, event)
                except ValueError as exc:
                    await websocket.send_json({"type": "error", "detail": str(exc)})
                    continue
                record = await _send_probe_for_segment(websocket, interview_id, record, segment)
            elif event.get("type") == "end":
                result = end_interview(interview_id)
                event_type = "task_queued" if isinstance(result, OfflineTaskAccepted) else "report"
                await websocket.send_json({"type": event_type, "payload": result.model_dump(mode="json")})
                asr_session_manager.close(interview_id)
                break
    except WebSocketDisconnect:
        asr_session_manager.close(interview_id)
        return
    except KeyError as exc:
        asr_session_manager.close(interview_id)
        await websocket.send_json({"type": "error", "detail": str(exc)})
    except ValueError as exc:
        asr_session_manager.close(interview_id)
        await websocket.send_json({"type": "error", "detail": str(exc)})
