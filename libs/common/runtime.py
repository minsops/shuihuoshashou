from __future__ import annotations

from pydantic import BaseModel

from libs.common.config import get_settings


class RuntimeStatus(BaseModel):
    app_env: str
    gateway_auth_enabled: bool
    database_url: str
    llm_provider: str
    llm_model: str
    llm_base_url_configured: bool
    llm_api_path: str
    llm_api_key_configured: bool
    llm_auth_header: str
    llm_auth_scheme_configured: bool
    llm_response_content_path: str
    llm_extra_body_configured: bool
    llm_max_retries: int
    asr_provider: str
    asr_base_url_configured: bool
    asr_api_key_configured: bool
    asr_channel_diarization_configured: bool
    probe_min_answer_chars: int
    probe_min_interval_ms: int
    speaker_diarization_provider: str
    speaker_diarization_base_url_configured: bool
    speaker_diarization_api_key_configured: bool
    signal_enabled: bool
    rate_limit_enabled: bool
    rate_limit_backend: str
    rate_limit_requests_per_minute: int
    redis_rate_limit_prefix: str
    offline_task_backend: str
    offline_task_execution: str
    celery_broker_configured: bool
    celery_result_backend_configured: bool
    redis_url_configured: bool
    redis_stream_prefix: str
    jd_vector_backend: str
    object_storage_endpoint_configured: bool
    object_storage_credentials_configured: bool
    object_storage_bucket: str
    report_dir: str


def get_runtime_status() -> RuntimeStatus:
    settings = get_settings()
    return RuntimeStatus(
        app_env=settings.app_env,
        gateway_auth_enabled=bool(settings.gateway_api_key),
        database_url=settings.database_url,
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        llm_base_url_configured=bool(settings.llm_base_url),
        llm_api_path=settings.llm_api_path,
        llm_api_key_configured=bool(settings.llm_api_key),
        llm_auth_header=settings.llm_auth_header,
        llm_auth_scheme_configured=bool(settings.llm_auth_scheme),
        llm_response_content_path=settings.llm_response_content_path,
        llm_extra_body_configured=bool(settings.llm_extra_body_json),
        llm_max_retries=settings.llm_max_retries,
        asr_provider=settings.asr_provider,
        asr_base_url_configured=bool(settings.asr_base_url),
        asr_api_key_configured=bool(settings.asr_api_key),
        asr_channel_diarization_configured=bool(
            settings.asr_interviewer_channels and settings.asr_candidate_channels
        ),
        probe_min_answer_chars=settings.probe_min_answer_chars,
        probe_min_interval_ms=settings.probe_min_interval_ms,
        speaker_diarization_provider=settings.speaker_diarization_provider,
        speaker_diarization_base_url_configured=bool(settings.speaker_diarization_base_url),
        speaker_diarization_api_key_configured=bool(settings.speaker_diarization_api_key),
        signal_enabled=settings.signal_enabled,
        rate_limit_enabled=settings.rate_limit_enabled,
        rate_limit_backend=settings.rate_limit_backend,
        rate_limit_requests_per_minute=settings.rate_limit_requests_per_minute,
        redis_rate_limit_prefix=settings.redis_rate_limit_prefix,
        offline_task_backend=settings.offline_task_backend,
        offline_task_execution=settings.offline_task_execution,
        celery_broker_configured=bool(settings.celery_broker_url),
        celery_result_backend_configured=bool(settings.celery_result_backend),
        redis_url_configured=bool(settings.redis_url),
        redis_stream_prefix=settings.redis_stream_prefix,
        jd_vector_backend=settings.jd_vector_backend,
        object_storage_endpoint_configured=bool(settings.object_storage_endpoint),
        object_storage_credentials_configured=bool(
            settings.object_storage_access_key and settings.object_storage_secret_key
        ),
        object_storage_bucket=settings.object_storage_bucket,
        report_dir=str(settings.report_dir),
    )
