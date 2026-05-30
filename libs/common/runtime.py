from __future__ import annotations

from pydantic import BaseModel

from libs.common.config import get_settings


class RuntimeStatus(BaseModel):
    app_env: str
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
    signal_enabled: bool
    rate_limit_enabled: bool
    rate_limit_requests_per_minute: int
    offline_task_backend: str
    redis_url_configured: bool
    object_storage_endpoint_configured: bool
    object_storage_bucket: str
    report_dir: str


def get_runtime_status() -> RuntimeStatus:
    settings = get_settings()
    return RuntimeStatus(
        app_env=settings.app_env,
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
        signal_enabled=settings.signal_enabled,
        rate_limit_enabled=settings.rate_limit_enabled,
        rate_limit_requests_per_minute=settings.rate_limit_requests_per_minute,
        offline_task_backend=settings.offline_task_backend,
        redis_url_configured=bool(settings.redis_url),
        object_storage_endpoint_configured=bool(settings.object_storage_endpoint),
        object_storage_bucket=settings.object_storage_bucket,
        report_dir=str(settings.report_dir),
    )
