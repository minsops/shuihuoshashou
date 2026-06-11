from __future__ import annotations

import asyncio
import base64
import binascii
import json
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import httpx
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response
from starlette.datastructures import Headers

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
    CredibilitySignal,
    InterviewCreate,
    JobCreate,
    OfflineInterviewInput,
    OfflineInterviewResult,
    ProbeResponse,
    OfflineTaskAccepted,
    ProbeRequest,
    ProbeSuggestion,
    QATurn,
    ResumeClaim,
    ReportBuildRequest,
    ScoringRequest,
    TranscriptSegment,
)
from libs.llm_client import LLMMessage, get_llm_client
from services.asr_service.nls_token import AliyunNLSTokenProvider
from services.aigc_detect_service.service import detect_interview
from services.asr_service.service import (
    ASREngine,
    asr_session_manager,
    configure_asr_runtime,
    get_asr_engine,
)
from services.document_service import parse_document
from services.interview_orchestrator.dialogue import DialogueAssembler
from services.interview_orchestrator.service import (
    add_turn,
    add_utterance,
    create_candidate,
    create_consent,
    create_interview,
    end_interview,
    get_interview,
    get_report,
    has_active_consent,
    should_probe_turn,
    start_interview,
)
from services.jd_kb_service.service import create_job, get_job, retrieve_job_probe_patterns
from services.probe_service.service import fallback_probe, generate_probe
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
ALIYUN_AUDIO_CONTEXT_LIMIT = 200
STREAMING_ASR_PROVIDERS = {"aliyun_ws", "aliyun_nls_ws"}
LOCAL_DEV_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$"


class _LockedWebSocketSender:
    def __init__(self, websocket: WebSocket) -> None:
        self._websocket = websocket
        self._lock = asyncio.Lock()

    async def send_json(self, payload: dict) -> None:
        async with self._lock:
            await self._websocket.send_json(payload)


@dataclass(frozen=True)
class _AliyunAudioContext:
    seq: int
    start_ms: int
    end_ms: int
    audio_b64: str
    speaker_hint: str | None
    question: str
    question_source: str
    probe_target: str | None
    probe_chain_id: str | None


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    configure_opentelemetry(fastapi_app, get_settings())
    configure_asr_runtime()
    init_db()
    yield


app = FastAPI(title="Shuihuo Killer", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=LOCAL_DEV_ORIGIN_REGEX,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
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
    websocket_key = _extract_websocket_protocol_key(headers)
    if websocket_key:
        return websocket_key
    return query_key or ""


def _extract_websocket_protocol_key(headers: Headers) -> str:
    protocols = headers.get("sec-websocket-protocol", "")
    for protocol in protocols.split(","):
        item = protocol.strip()
        if not item.startswith("gateway-key."):
            continue
        encoded = item.removeprefix("gateway-key.")
        try:
            padding = "=" * (-len(encoded) % 4)
            return base64.urlsafe_b64decode(f"{encoded}{padding}").decode("utf-8")
        except (binascii.Error, UnicodeDecodeError):
            return ""
    return ""


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
    return {
        "status": "ok",
        "service": "shuihuo-killer-gateway",
        "version": app.version,
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    return metrics_registry.render_prometheus()


@app.get("/api/config/status", response_model=RuntimeStatus)
def config_status() -> RuntimeStatus:
    return get_runtime_status()


@app.post("/api/config/llm/check")
async def check_llm_connection() -> dict[str, str | bool]:
    status = get_runtime_status()
    if status.llm_provider == "mock":
        return {
            "ok": False,
            "mode": "mock",
            "message": "当前是模型模拟模式，没有调用真实模型。",
        }
    if not status.llm_base_url_configured or not status.llm_api_key_configured:
        return {
            "ok": False,
            "mode": "incomplete",
            "message": "模型配置不完整，请检查 LLM_BASE_URL 和 LLM_API_KEY。",
        }
    fallback = ProbeResponse(
        suggestions=[
            ProbeSuggestion(
                question="fallback",
                target="fallback",
                competency="fallback",
                priority=1,
            )
        ],
        credibility=CredibilitySignal(level="vague", reason="fallback", drill_down_hint="fallback"),
    )
    try:
        response = await get_llm_client().complete_json(
            [
                LLMMessage(
                    role="system",
                    content=(
                        "Return only a JSON object with this exact shape: "
                        '{"suggestions":[{"question":"...","target":"...","competency":"...",'
                        '"priority":1}],"credibility":{"level":"solid","reason":"...",'
                        '"drill_down_hint":"..."}}. The credibility.level must be one of '
                        "solid, vague, suspicious."
                    ),
                ),
                LLMMessage(
                    role="user",
                    content="Generate one concise interview probe question for an LLM connection check.",
                ),
            ],
            ProbeResponse,
            fallback,
            raise_on_error=True,
        )
    except RuntimeError as exc:
        return {"ok": False, "mode": "error", "message": _safe_public_error(str(exc))}
    if response == fallback:
        return {
            "ok": False,
            "mode": "fallback",
            "message": "真实模型调用未得到有效 JSON 响应，请检查 provider 协议和响应字段路径。",
        }
    return {
        "ok": True,
        "mode": "live",
        "message": f"真实模型连接正常：{status.llm_provider} / {status.llm_model}",
    }


def _safe_public_error(message: str) -> str:
    cleaned = " ".join(message.split())
    for secret in _configured_secret_values():
        cleaned = cleaned.replace(secret, "***")
    if "HTTP " in cleaned:
        return cleaned[cleaned.index("HTTP ") :].split(":", maxsplit=1)[0]
    return cleaned[:240] or "unknown LLM error"


def _configured_secret_values() -> list[str]:
    try:
        settings = get_settings()
    except Exception:
        return []
    return [
        secret
        for secret in {
            settings.llm_api_key,
            settings.asr_api_key,
            settings.aliyun_asr_api_key,
            settings.aliyun_nls_token,
            settings.aliyun_ak_secret,
            settings.speaker_diarization_api_key,
            settings.aigc_detector_api_key,
            settings.gateway_api_key,
        }
        if len(secret.strip()) >= 4
    ]


@app.post("/api/config/asr/check")
async def check_asr_readiness() -> dict[str, str | bool]:
    settings = get_settings()
    status = get_runtime_status()
    if status.asr_provider == "stub":
        return {
            "ok": False,
            "mode": "stub",
            "message": "当前是 ASR 模拟模式，没有调用真实语音识别。",
        }
    if status.asr_provider == "http":
        if not status.asr_base_url_configured:
            return {
                "ok": False,
                "mode": "incomplete",
                "message": "HTTP ASR 配置不完整，请检查 ASR_BASE_URL。",
            }
        return {
            "ok": True,
            "mode": "configured",
            "message": "HTTP ASR 基本配置存在；此检查未发送测试音频。",
        }
    if status.asr_provider == "aliyun_ws":
        try:
            import websockets  # noqa: F401
        except ImportError:
            return {
                "ok": False,
                "mode": "missing_dependency",
                "message": "ASR WebSocket 依赖缺失，请重新安装依赖。",
            }
        if not status.aliyun_asr_api_key_configured or not status.aliyun_asr_endpoint_configured:
            return {
                "ok": False,
                "mode": "incomplete",
                "message": "阿里云 DashScope ASR 配置不完整，请检查 API Key 和 Endpoint。",
            }
        return {
            "ok": True,
            "mode": "configured",
            "message": "阿里云 DashScope ASR 基本配置存在；此检查未打开识别会话。",
        }
    if status.asr_provider == "aliyun_nls_ws":
        try:
            import websockets  # noqa: F401
        except ImportError:
            return {
                "ok": False,
                "mode": "missing_dependency",
                "message": "ASR WebSocket 依赖缺失，请重新安装依赖。",
            }
        if not status.aliyun_nls_app_key_configured or not status.aliyun_nls_endpoint_configured:
            return {
                "ok": False,
                "mode": "incomplete",
                "message": "阿里云 NLS ASR 配置不完整，请检查 AppKey 和 Endpoint。",
            }
        if status.aliyun_nls_token_configured:
            if status.aliyun_nls_token_auto_configured:
                try:
                    token = await asyncio.to_thread(
                        AliyunNLSTokenProvider(
                            access_key_id=settings.aliyun_ak_id,
                            access_key_secret=settings.aliyun_ak_secret,
                            endpoint=settings.aliyun_nls_token_endpoint,
                            region_id=settings.aliyun_nls_token_region,
                        ).get_token
                    )
                except Exception as exc:
                    return {
                        "ok": True,
                        "mode": "fixed_token",
                        "message": _safe_public_error(
                            f"阿里云 NLS ASR 固定 Token 已配置；AK 自动 Token 创建失败: {exc}"
                        ),
                    }
                if token.strip():
                    return {
                        "ok": True,
                        "mode": "fixed_token_auto_ready",
                        "message": "阿里云 NLS ASR 就绪：固定 Token 已配置，AK 自动 Token 可生成；固定 Token 被拒绝时会自动刷新重试。",
                    }
            return {
                "ok": True,
                "mode": "fixed_token",
                "message": "阿里云 NLS ASR 基本配置存在：固定 Token 已配置；此轻量检查不打开音频识别会话。",
            }
        if status.aliyun_nls_token_auto_configured:
            try:
                token = await asyncio.to_thread(
                    AliyunNLSTokenProvider(
                        access_key_id=settings.aliyun_ak_id,
                        access_key_secret=settings.aliyun_ak_secret,
                        endpoint=settings.aliyun_nls_token_endpoint,
                        region_id=settings.aliyun_nls_token_region,
                    ).get_token
                )
            except Exception as exc:
                return {
                    "ok": False,
                    "mode": "token_error",
                    "message": _safe_public_error(f"NLS Token 创建失败: {exc}"),
                }
            if token.strip():
                return {
                    "ok": True,
                    "mode": "token_ready",
                    "message": "阿里云 NLS ASR 就绪：WebSocket 依赖存在，自动 Token 可生成。",
                }
        return {
            "ok": False,
            "mode": "incomplete",
            "message": "阿里云 NLS ASR 缺少 Token，请配置固定 Token 或 AK 自动 Token。",
        }
    return {"ok": False, "mode": "unknown", "message": "ASR provider 状态未知。"}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return WEB_INDEX.read_text(encoding="utf-8")


def _probe_request_for_turn(record, turn: QATurn) -> ProbeRequest:
    return ProbeRequest(
        job_id=record.job_id,
        competency_model=record.context.competency_model,
        recent_turns=record.context.turns[-5:],
        latest_answer=turn.answer,
        resume_claims=_resume_claims_for_probe(record),
        probe_chains=record.context.probe_chains,
    )


def _resume_claims_for_probe(record) -> list[ResumeClaim]:
    markers = ("独立", "主导", "负责", "提升", "优化", "上线", "%", "qps", "ms", "指标")
    claims: list[ResumeClaim] = []
    for raw in re.split(r"[。\n；;]", record.context.candidate_resume_text):
        text = raw.strip()
        if len(text) < 8 or not any(marker in text for marker in markers):
            continue
        tags = [marker for marker in markers if marker in text][:4]
        claims.append(ResumeClaim(text=text, tags=tags or ["resume"]))
        if len(claims) >= 5:
            break
    return claims


def _schedule_probe_task(
    websocket: WebSocket,
    record,
    turn: QATurn,
) -> None:
    request = _probe_request_for_turn(record, turn)

    async def _run() -> None:
        try:
            fallback = fallback_probe(request)
        except Exception:
            return
        await websocket.send_json({"type": "probe", "payload": fallback.model_dump()})
        await websocket.send_json(
            {"type": "credibility", "payload": fallback.credibility.model_dump()}
        )
        signal = (
            extract_behavior_signal(turn)
            if record.signal_enabled and has_active_consent(record.candidate_id, "behavior_signal")
            else None
        )
        if signal is not None:
            await websocket.send_json({"type": "signal", "payload": signal.model_dump()})
        try:
            probe = await generate_probe(request)
        except Exception:
            return
        if probe.model_dump() != fallback.model_dump():
            await websocket.send_json({"type": "probe_update", "payload": probe.model_dump()})

    asyncio.create_task(_run())


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


def _event_question(event: dict) -> str:
    return str(event.get("question") or "实时输入片段").strip() or "实时输入片段"


def _event_question_source(event: dict) -> str:
    return "ai_probe" if str(event.get("question_source") or "").strip() == "ai_probe" else "interviewer"


def _event_probe_target(event: dict) -> str | None:
    value = str(event.get("probe_target") or "").strip()
    return value or None


def _event_probe_chain_id(event: dict) -> str | None:
    value = str(event.get("chain_id") or event.get("probe_chain_id") or "").strip()
    return value or None


def _aliyun_warning_reason(exc: Exception) -> str:
    reason = str(exc).strip()
    return reason if reason.startswith("aliyun_asr_") else "aliyun_asr_connect_failed"


def _asr_warning_reason(exc: Exception) -> str:
    reason = str(exc).strip()
    if reason.startswith("aliyun_asr_"):
        return reason
    return "asr_transcription_failed"


def _select_aliyun_audio_context(
    contexts: list[_AliyunAudioContext],
    segment: TranscriptSegment,
) -> _AliyunAudioContext | None:
    if not contexts:
        return None

    def score(context: _AliyunAudioContext) -> tuple[int, int, int]:
        overlap = min(context.end_ms, segment.end_ms) - max(context.start_ms, segment.start_ms)
        if overlap >= 0:
            return (2, overlap, context.seq)
        if context.end_ms <= segment.start_ms:
            return (1, -abs(segment.start_ms - context.end_ms), context.seq)
        return (0, -abs(context.start_ms - segment.end_ms), context.seq)

    return max(contexts, key=score)


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
        await _send_asr_warning(websocket, _asr_warning_reason(exc), seq)
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


def _event_timestamp_bounds(event: dict, seq: int) -> tuple[int, int]:
    start_ms = _event_int(event, "start_ms")
    end_ms = _event_int(event, "end_ms")
    start = max(0, start_ms if start_ms is not None else seq * 1000)
    end = end_ms if end_ms is not None else start + 900
    return start, max(start, end)


def _event_confidence(event: dict, default: float) -> float:
    value = event.get("confidence", default)
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, confidence))


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


@app.post("/api/documents/parse")
async def api_parse_document(request: Request, kind: str = "resume", filename: str = "upload"):
    if kind not in {"jd", "resume"}:
        raise HTTPException(status_code=400, detail="kind must be jd or resume")
    data = await request.body()
    try:
        return parse_document(
            filename,
            data,
            kind=kind,  # type: ignore[arg-type]
            content_type=request.headers.get("content-type", ""),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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


@app.get("/api/interviews/{interview_id}")
def api_get_interview(interview_id: str):
    try:
        return get_interview(interview_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


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
            {
                "qa_turns": transcript,
                "full_transcript": report.get("utterances", []),
                "probe_chains": report.get("probe_chains", []),
                "analysis_mode": report.get("analysis_mode", "llm"),
            },
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
    sender = _LockedWebSocketSender(websocket)
    engine = get_asr_engine()
    aliyun_session = None
    aliyun_reader_task: asyncio.Task[None] | None = None
    aliyun_result_seq = 1
    aliyun_audio_contexts: list[_AliyunAudioContext] = []
    assembler = DialogueAssembler(silence_close_ms=settings.dialogue_silence_close_ms)
    dialogue_lock = asyncio.Lock()
    dialogue_epoch = 0
    dialogue_flush_task: asyncio.Task[None] | None = None

    def cancel_dialogue_silence_flush() -> None:
        nonlocal dialogue_flush_task
        if dialogue_flush_task is not None and not dialogue_flush_task.done():
            dialogue_flush_task.cancel()
        dialogue_flush_task = None

    def schedule_dialogue_silence_flush() -> None:
        nonlocal dialogue_epoch, dialogue_flush_task
        if settings.dialogue_silence_close_ms < 0:
            return
        dialogue_epoch += 1
        scheduled_epoch = dialogue_epoch
        cancel_dialogue_silence_flush()

        async def _flush_after_silence() -> None:
            try:
                await asyncio.sleep(settings.dialogue_silence_close_ms / 1000)
                async with dialogue_lock:
                    if scheduled_epoch != dialogue_epoch:
                        return
                    await persist_dialogue_result(assembler.flush())
            except asyncio.CancelledError:
                return

        dialogue_flush_task = asyncio.create_task(_flush_after_silence())

    async def handle_dialogue_segment(
        segment: TranscriptSegment,
        *,
        fallback_question: str | None = None,
        question_source: str = "interviewer",
        probe_target: str | None = None,
        probe_chain_id: str | None = None,
        force_close: bool = False,
        emit_probe: bool = True,
        force_probe: bool = False,
    ) -> None:
        nonlocal dialogue_epoch
        async with dialogue_lock:
            if segment.is_final:
                dialogue_epoch += 1
                cancel_dialogue_silence_flush()
            result = assembler.feed(
                segment,
                fallback_question=fallback_question,
                question_source=question_source,
                probe_target=probe_target,
                probe_chain_id=probe_chain_id,
                force_close=force_close,
            )
            await persist_dialogue_result(result, emit_probe=emit_probe, force_probe=force_probe)
            if segment.is_final and not force_close:
                schedule_dialogue_silence_flush()

    async def persist_dialogue_result(
        result,
        *,
        emit_probe: bool = True,
        force_probe: bool = False,
    ) -> None:
        nonlocal record
        for utterance in result.utterances:
            record = add_utterance(interview_id, utterance)
        for turn in result.turns:
            record = add_turn(interview_id, turn)
            await sender.send_json(
                {
                    "type": "probe_chains",
                    "payload": [chain.model_dump() for chain in record.context.probe_chains],
                }
            )
            if emit_probe and (force_probe or should_probe_turn(turn, record)):
                _schedule_probe_task(sender, record, turn)

    async def ensure_aliyun_session():
        nonlocal aliyun_session, aliyun_reader_task
        if settings.asr_provider not in STREAMING_ASR_PROVIDERS:
            return None
        if aliyun_session is not None and not getattr(aliyun_session, "finished", False):
            return aliyun_session
        try:
            aliyun_session = await engine.get_or_create_session(interview_id)  # type: ignore[attr-defined]
        except Exception as exc:
            await _send_asr_warning(sender, _aliyun_warning_reason(exc), 0)
            return None
        aliyun_reader_task = asyncio.create_task(_aliyun_result_reader(aliyun_session))
        return aliyun_session

    async def _aliyun_result_reader(session) -> None:
        nonlocal record, aliyun_result_seq
        while True:
            item = await session.result_queue.get()
            if item is None:
                if session.error_reason:
                    await _send_asr_warning(sender, session.error_reason, aliyun_result_seq)
                return
            seq = aliyun_result_seq
            aliyun_result_seq += 1
            context = _select_aliyun_audio_context(aliyun_audio_contexts, item)
            if context is not None and context.speaker_hint is not None and item.speaker == "unknown":
                item = item.model_copy(update={"speaker": context.speaker_hint})
            decision = asr_session_manager.accept_segment(
                seq,
                item,
                audio_b64=context.audio_b64 if context is not None else None,
            )
            if not decision.accepted or decision.segment is None:
                await _send_asr_warning(sender, decision.reason, seq)
                continue
            segment = decision.segment
            await sender.send_json({"type": "transcript", "payload": segment.model_dump()})
            await handle_dialogue_segment(
                segment,
                fallback_question=context.question if context is not None else "实时语音片段",
                question_source=context.question_source if context is not None else "interviewer",
                probe_target=context.probe_target if context is not None else None,
                probe_chain_id=context.probe_chain_id if context is not None else None,
            )

    try:
        record = start_interview(interview_id)
        while True:
            try:
                event = await websocket.receive_json()
            except json.JSONDecodeError:
                await sender.send_json(
                    {"type": "error", "detail": "event payload must be valid JSON"}
                )
                continue
            except KeyError:
                await sender.send_json(
                    {"type": "error", "detail": "event payload must be a text JSON frame"}
                )
                continue
            if not isinstance(event, dict):
                await sender.send_json(
                    {"type": "error", "detail": "event payload must be an object"}
                )
                continue
            if event.get("type") == "audio_chunk":
                seq = _event_seq(event, 0)
                if seq is None:
                    await _send_asr_warning(sender, "invalid_seq", 0)
                    continue
                if _event_session_mismatch(interview_id, event):
                    await _send_asr_warning(sender, "session_id_mismatch", seq)
                    continue
                audio_b64 = event.get("audio", "")
                if not _valid_audio_b64(audio_b64):
                    await _send_asr_warning(sender, "invalid_audio_base64", seq)
                    continue
                contract_warning = _audio_contract_warning(event)
                if contract_warning is not None:
                    await _send_asr_warning(sender, contract_warning, seq)
                    continue
                if settings.asr_provider in STREAMING_ASR_PROVIDERS:
                    session = await ensure_aliyun_session()
                    if session is None:
                        continue
                    start_ms, end_ms = _event_timestamp_bounds(event, seq)
                    aliyun_audio_contexts.append(
                        _AliyunAudioContext(
                            seq=seq,
                            start_ms=start_ms,
                            end_ms=end_ms,
                            audio_b64=audio_b64,
                            speaker_hint=_event_speaker(event),
                            question=_event_question(event),
                            question_source=_event_question_source(event),
                            probe_target=_event_probe_target(event),
                            probe_chain_id=_event_probe_chain_id(event),
                        )
                    )
                    if len(aliyun_audio_contexts) > ALIYUN_AUDIO_CONTEXT_LIMIT:
                        del aliyun_audio_contexts[:-ALIYUN_AUDIO_CONTEXT_LIMIT]
                    pcm_bytes = base64.b64decode(audio_b64)
                    try:
                        await session.send_audio(pcm_bytes)
                    except Exception:
                        if aliyun_session is not None:
                            await aliyun_session.close()
                        if hasattr(engine, "close_session"):
                            await engine.close_session(interview_id)  # type: ignore[attr-defined]
                        aliyun_session = None
                        session = await ensure_aliyun_session()
                        if session is None:
                            continue
                        try:
                            await session.send_audio(pcm_bytes)
                        except Exception as exc:
                            await _send_asr_warning(sender, _asr_warning_reason(exc), seq)
                    continue
                segment = await _transcribe_or_warn(
                    sender,
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
                    await _send_asr_warning(sender, decision.reason, seq)
                    continue
                segment = decision.segment
                await sender.send_json({"type": "transcript", "payload": segment.model_dump()})
                await handle_dialogue_segment(
                    segment,
                    fallback_question=_event_question(event),
                    question_source=_event_question_source(event),
                    probe_target=_event_probe_target(event),
                    probe_chain_id=_event_probe_chain_id(event),
                )
            elif event.get("type") == "text_turn":
                seq = _event_seq(event, 1)
                if seq is None:
                    await sender.send_json({"type": "error", "detail": "invalid seq"})
                    continue
                text = str(event.get("answer", "")).strip()
                if not text:
                    await sender.send_json(
                        {"type": "error", "detail": "text_turn requires answer"}
                    )
                    continue
                encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
                if settings.asr_provider in STREAMING_ASR_PROVIDERS:
                    start_ms, end_ms = _event_timestamp_bounds(event, seq)
                    segment = TranscriptSegment(
                        session_id=interview_id,
                        speaker=_event_speaker(event) or "candidate",
                        text=text,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        is_final=_event_bool(event, "is_final", True),
                        confidence=_event_confidence(event, 1.0),
                    )
                else:
                    segment = await _transcribe_or_warn(
                        sender,
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
                    await _send_asr_warning(sender, decision.reason, seq)
                    continue
                segment = decision.segment
                await sender.send_json({"type": "transcript", "payload": segment.model_dump()})
                await handle_dialogue_segment(
                    segment,
                    fallback_question=_event_question(event),
                    question_source=_event_question_source(event),
                    probe_target=_event_probe_target(event),
                    probe_chain_id=_event_probe_chain_id(event),
                    force_close=True,
                )
            elif event.get("type") == "manual_probe":
                if not str(event.get("answer") or event.get("latest_answer") or "").strip():
                    await sender.send_json(
                        {"type": "error", "detail": "manual_probe requires answer"}
                    )
                    continue
                try:
                    segment = _manual_probe_segment(interview_id, event)
                except ValueError as exc:
                    await sender.send_json({"type": "error", "detail": str(exc)})
                    continue
                await sender.send_json({"type": "transcript", "payload": segment.model_dump()})
                await handle_dialogue_segment(
                    segment,
                    fallback_question=_event_question(event),
                    question_source=_event_question_source(event),
                    probe_target=_event_probe_target(event),
                    probe_chain_id=_event_probe_chain_id(event),
                    force_close=True,
                    force_probe=True,
                )
            elif event.get("type") == "end":
                try:
                    cancel_dialogue_silence_flush()
                    if aliyun_session is not None:
                        await aliyun_session.close()
                    if aliyun_reader_task is not None:
                        await aliyun_reader_task
                    await persist_dialogue_result(assembler.flush(), emit_probe=False)
                    result = await asyncio.to_thread(end_interview, interview_id)
                except (KeyError, ValueError) as exc:
                    await sender.send_json({"type": "error", "detail": str(exc)})
                    continue
                event_type = "task_queued" if isinstance(result, OfflineTaskAccepted) else "report"
                await sender.send_json({"type": event_type, "payload": result.model_dump(mode="json")})
                asr_session_manager.close(interview_id)
                break
            else:
                await sender.send_json({"type": "error", "detail": "unsupported event type"})
    except WebSocketDisconnect:
        cancel_dialogue_silence_flush()
        if aliyun_session is not None:
            await aliyun_session.close()
        asr_session_manager.close(interview_id)
        return
    except KeyError as exc:
        cancel_dialogue_silence_flush()
        if aliyun_session is not None:
            await aliyun_session.close()
        asr_session_manager.close(interview_id)
        await sender.send_json({"type": "error", "detail": str(exc)})
    except ValueError as exc:
        cancel_dialogue_silence_flush()
        if aliyun_session is not None:
            await aliyun_session.close()
        asr_session_manager.close(interview_id)
        await sender.send_json({"type": "error", "detail": str(exc)})
