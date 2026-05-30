from __future__ import annotations

import base64
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from starlette.datastructures import Headers
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, Response

from libs.common.config import get_settings
from libs.common.database import init_db
from libs.common.observability import (
    log_event,
    metrics_registry,
    rate_limiter,
    request_id_from_header,
    trace_context_from_header,
)
from libs.common.runtime import RuntimeStatus, get_runtime_status
from libs.schemas import (
    CandidateCreate,
    ConsentCreate,
    InterviewCreate,
    JobCreate,
    OfflineInterviewInput,
    OfflineInterviewResult,
    ProbeRequest,
    QATurn,
)
from services.asr_service.service import asr_session_manager, get_asr_engine
from services.interview_orchestrator.service import (
    add_turn,
    create_candidate,
    create_consent,
    create_interview,
    end_interview,
    get_interview,
    get_report,
    should_probe,
    start_interview,
)
from services.jd_kb_service.service import create_job, get_job, retrieve_job_probe_patterns
from services.probe_service.service import generate_probe
from services.signal_service.service import extract_behavior_signal

VALID_SPEAKERS = {"interviewer", "candidate", "unknown"}


@asynccontextmanager
async def lifespan(_: FastAPI):
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
    rate_limiter.requests_per_minute = settings.rate_limit_requests_per_minute
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
        decision = rate_limiter.check(_client_key(request))
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
    signal = extract_behavior_signal(turn) if record.signal_enabled else None
    if signal is not None:
        await websocket.send_json({"type": "signal", "payload": signal.model_dump()})
    return record


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


def _channel_aliases(raw: str) -> set[str]:
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


def _event_bool(event: dict, key: str, default: bool) -> bool:
    value = event.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() not in {"0", "false", "no"}
    return bool(value)


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
    return create_consent(payload)


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


@app.post("/api/interviews/{interview_id}/turns")
def api_add_turn(interview_id: str, turn: QATurn):
    try:
        return add_turn(interview_id, turn)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/probe")
async def api_probe(payload: ProbeRequest):
    return await generate_probe(payload)


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


@app.get("/api/interviews/{interview_id}/report.pdf")
def api_report_pdf(interview_id: str):
    try:
        report, _ = get_report(interview_id)
        pdf_path = report.get("pdf_path")
        if not pdf_path:
            raise KeyError(f"report pdf not found: {interview_id}")
        return FileResponse(pdf_path, media_type="application/pdf", filename=f"{interview_id}.pdf")
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
                seq = int(event.get("seq", 0))
                segment = await engine.transcribe_chunk(
                    session_id=interview_id,
                    seq=seq,
                    audio_b64=event.get("audio", ""),
                    speaker=_event_speaker(event),
                    start_ms=event.get("start_ms"),
                    end_ms=event.get("end_ms"),
                    is_final=_event_bool(event, "is_final", True),
                    confidence=event.get("confidence"),
                )
                decision = asr_session_manager.accept_segment(
                    seq,
                    segment,
                    audio_b64=event.get("audio", ""),
                )
                if not decision.accepted or decision.segment is None:
                    await websocket.send_json(
                        {"type": "asr_warning", "payload": {"reason": decision.reason, "seq": seq}}
                    )
                    continue
                segment = decision.segment
                await websocket.send_json({"type": "transcript", "payload": segment.model_dump()})
                if should_probe(segment, record):
                    record = await _send_probe_for_segment(websocket, interview_id, record, segment)
            elif event.get("type") == "text_turn":
                seq = int(event.get("seq", 1))
                text = str(event.get("answer", ""))
                encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
                segment = await engine.transcribe_chunk(
                    interview_id,
                    seq,
                    encoded,
                    speaker=_event_speaker(event) or "candidate",
                    start_ms=event.get("start_ms"),
                    end_ms=event.get("end_ms"),
                    is_final=_event_bool(event, "is_final", True),
                    confidence=event.get("confidence"),
                )
                decision = asr_session_manager.accept_segment(seq, segment, audio_b64=encoded)
                if not decision.accepted or decision.segment is None:
                    await websocket.send_json(
                        {"type": "asr_warning", "payload": {"reason": decision.reason, "seq": seq}}
                    )
                    continue
                segment = decision.segment
                await websocket.send_json({"type": "transcript", "payload": segment.model_dump()})
                if should_probe(segment, record):
                    record = await _send_probe_for_segment(websocket, interview_id, record, segment)
            elif event.get("type") == "end":
                report = end_interview(interview_id)
                await websocket.send_json({"type": "report", "payload": report.model_dump(mode="json")})
                asr_session_manager.close(interview_id)
                break
    except WebSocketDisconnect:
        asr_session_manager.close(interview_id)
        return
    except KeyError as exc:
        asr_session_manager.close(interview_id)
        await websocket.send_json({"type": "error", "detail": str(exc)})
