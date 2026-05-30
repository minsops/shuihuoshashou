from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "local"
    gateway_api_key: str = ""
    database_url: str = "sqlite:///data/shuihuo_killer.db"
    llm_provider: str = "mock"
    llm_model: str = "mimo-v2.5-pro"
    llm_base_url: str = "https://token-plan-cn.xiaomimimo.com/v1"
    llm_api_path: str = "/chat/completions"
    llm_api_key: str = ""
    llm_auth_header: str = "api-key"
    llm_auth_scheme: str = ""
    llm_response_content_path: str = "choices.0.message.content"
    llm_extra_body_json: str = ""
    llm_timeout_seconds: int = 30
    llm_max_retries: int = 1
    asr_provider: str = "stub"
    asr_base_url: str = ""
    asr_api_path: str = "/transcribe"
    asr_api_key: str = ""
    asr_auth_header: str = "Authorization"
    asr_auth_scheme: str = "Bearer"
    asr_text_path: str = "text"
    asr_speaker_path: str = "speaker"
    asr_start_ms_path: str = "start_ms"
    asr_end_ms_path: str = "end_ms"
    asr_is_final_path: str = "is_final"
    asr_confidence_path: str = "confidence"
    asr_timeout_seconds: int = 30
    asr_interviewer_channels: str = "0,left,interviewer"
    asr_candidate_channels: str = "1,right,candidate"
    probe_min_answer_chars: int = 20
    probe_min_interval_ms: int = 1000
    speaker_diarization_provider: str = "local"
    speaker_diarization_base_url: str = ""
    speaker_diarization_api_path: str = "/diarize"
    speaker_diarization_api_key: str = ""
    speaker_diarization_auth_header: str = "Authorization"
    speaker_diarization_auth_scheme: str = "Bearer"
    speaker_diarization_speaker_path: str = "speaker"
    speaker_diarization_timeout_seconds: int = 10
    signal_enabled: bool = False
    rate_limit_enabled: bool = False
    rate_limit_backend: str = "local"
    rate_limit_requests_per_minute: int = 120
    redis_rate_limit_prefix: str = "shuihuo:rate_limit"
    offline_task_backend: str = "local"
    offline_task_execution: str = "sync"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"
    redis_url: str = "redis://localhost:6379/0"
    redis_stream_prefix: str = "shuihuo"
    jd_vector_backend: str = "local"
    object_storage_endpoint: str = ""
    object_storage_bucket: str = "shuihuo-killer"
    object_storage_access_key: str = ""
    object_storage_secret_key: str = ""
    object_storage_region: str = "us-east-1"
    report_dir: Path = Field(default=Path("data/reports"))

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
