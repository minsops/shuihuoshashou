from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "local"
    gateway_api_key: str = ""
    database_url: str = "sqlite:///data/shuihuo_killer.db"
    otel_exporter_otlp_endpoint: str = ""
    otel_service_name: str = "shuihuo-killer-gateway"
    llm_provider: Literal["mock", "openai_compatible"] = "mock"
    llm_model: str = "mimo-v2.5-pro"
    llm_base_url: str = "https://token-plan-cn.xiaomimimo.com/v1"
    llm_api_path: str = "/chat/completions"
    llm_api_key: str = ""
    llm_auth_header: str = "api-key"
    llm_auth_scheme: str = ""
    llm_response_content_path: str = "choices.0.message.content"
    llm_extra_body_json: str = ""
    llm_timeout_seconds: int = Field(default=30, gt=0)
    llm_max_retries: int = Field(default=1, ge=0)
    llm_rate_limit_enabled: bool = False
    llm_rate_limit_requests_per_minute: int = Field(default=60, ge=0)
    asr_provider: Literal["stub", "http"] = "stub"
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
    asr_timeout_seconds: int = Field(default=30, gt=0)
    asr_interviewer_channels: str = "0,left,interviewer"
    asr_candidate_channels: str = "1,right,candidate"
    probe_min_answer_chars: int = Field(default=20, ge=0)
    probe_min_interval_ms: int = Field(default=1000, ge=0)
    probe_require_topic_match: bool = True
    probe_topic_keywords: str = (
        "项目,技术,架构,方案,决策,优化,性能,指标,数据,负责,实现,"
        "上线,故障,异常,成本,延迟,吞吐,FastAPI,LLM,Python"
    )
    speaker_diarization_provider: Literal["local", "http"] = "local"
    speaker_diarization_base_url: str = ""
    speaker_diarization_api_path: str = "/diarize"
    speaker_diarization_api_key: str = ""
    speaker_diarization_auth_header: str = "Authorization"
    speaker_diarization_auth_scheme: str = "Bearer"
    speaker_diarization_speaker_path: str = "speaker"
    speaker_diarization_timeout_seconds: int = Field(default=10, gt=0)
    aigc_detector_provider: Literal["local", "http"] = "local"
    aigc_detector_base_url: str = ""
    aigc_detector_api_path: str = "/detect"
    aigc_detector_api_key: str = ""
    aigc_detector_auth_header: str = "Authorization"
    aigc_detector_auth_scheme: str = "Bearer"
    aigc_detector_probability_path: str = "ai_generated_prob"
    aigc_detector_flagged_path: str = "flagged"
    aigc_detector_timeout_seconds: int = Field(default=10, gt=0)
    aigc_ai_prob_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    aigc_template_similarity_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    signal_enabled: bool = False
    rate_limit_enabled: bool = False
    rate_limit_backend: Literal["local", "redis"] = "local"
    rate_limit_requests_per_minute: int = Field(default=120, ge=0)
    redis_rate_limit_prefix: str = "shuihuo:rate_limit"
    offline_task_backend: Literal["local", "redis_stream", "celery"] = "local"
    offline_task_execution: Literal["sync", "async"] = "sync"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"
    celery_task_queue: str = "shuihuo-offline"
    redis_url: str = "redis://localhost:6379/0"
    redis_stream_prefix: str = "shuihuo"
    jd_vector_backend: Literal["local", "pgvector"] = "local"
    object_storage_endpoint: str = ""
    object_storage_bucket: str = "shuihuo-killer"
    object_storage_access_key: str = ""
    object_storage_secret_key: str = ""
    object_storage_region: str = "us-east-1"
    report_dir: Path = Field(default=Path("data/reports"))

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @model_validator(mode="after")
    def runtime_dependencies_are_configured(self) -> "Settings":
        if self.asr_provider == "http" and not self.asr_base_url.strip():
            raise ValueError("ASR_PROVIDER=http requires ASR_BASE_URL")
        if (
            self.speaker_diarization_provider == "http"
            and not self.speaker_diarization_base_url.strip()
        ):
            raise ValueError(
                "SPEAKER_DIARIZATION_PROVIDER=http requires SPEAKER_DIARIZATION_BASE_URL"
            )
        if self.aigc_detector_provider == "http" and not self.aigc_detector_base_url.strip():
            raise ValueError("AIGC_DETECTOR_PROVIDER=http requires AIGC_DETECTOR_BASE_URL")
        if self.rate_limit_backend == "redis" and not self.redis_url.strip():
            raise ValueError("RATE_LIMIT_BACKEND=redis requires REDIS_URL")
        if self.offline_task_backend == "redis_stream" and not self.redis_url.strip():
            raise ValueError("OFFLINE_TASK_BACKEND=redis_stream requires REDIS_URL")
        if self.offline_task_backend == "celery":
            if not self.celery_broker_url.strip():
                raise ValueError("OFFLINE_TASK_BACKEND=celery requires CELERY_BROKER_URL")
            if not self.celery_result_backend.strip():
                raise ValueError("OFFLINE_TASK_BACKEND=celery requires CELERY_RESULT_BACKEND")
        if self.object_storage_endpoint and bool(self.object_storage_access_key) != bool(
            self.object_storage_secret_key
        ):
            raise ValueError(
                "OBJECT_STORAGE_ACCESS_KEY and OBJECT_STORAGE_SECRET_KEY must be configured together"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
