from __future__ import annotations

import pytest
from pydantic import ValidationError

from libs.common.config import Settings


def test_settings_rejects_unknown_runtime_backends() -> None:
    with pytest.raises(ValidationError):
        Settings(offline_task_backend="redis-stream")

    with pytest.raises(ValidationError):
        Settings(rate_limit_backend="memory")

    with pytest.raises(ValidationError):
        Settings(asr_provider="cloud")


def test_settings_rejects_out_of_range_runtime_thresholds() -> None:
    with pytest.raises(ValidationError):
        Settings(aigc_ai_prob_threshold=1.1)

    with pytest.raises(ValidationError):
        Settings(llm_timeout_seconds=0)

    with pytest.raises(ValidationError):
        Settings(probe_min_interval_ms=-1)


def test_settings_accepts_declared_production_backends() -> None:
    settings = Settings(
        llm_provider="openai_compatible",
        asr_provider="http",
        asr_base_url="https://asr.example.com",
        speaker_diarization_provider="http",
        speaker_diarization_base_url="https://diarize.example.com",
        aigc_detector_provider="http",
        aigc_detector_base_url="https://aigc.example.com",
        rate_limit_backend="redis",
        offline_task_backend="celery",
        offline_task_execution="async",
        jd_vector_backend="pgvector",
    )

    assert settings.llm_provider == "openai_compatible"
    assert settings.asr_provider == "http"
    assert settings.speaker_diarization_provider == "http"
    assert settings.aigc_detector_provider == "http"
    assert settings.rate_limit_backend == "redis"
    assert settings.offline_task_backend == "celery"
    assert settings.offline_task_execution == "async"
    assert settings.jd_vector_backend == "pgvector"


def test_settings_accepts_aliyun_ws_asr_provider() -> None:
    settings = Settings(asr_provider="aliyun_ws", aliyun_asr_api_key="dashscope-secret")

    assert settings.asr_provider == "aliyun_ws"
    assert settings.aliyun_asr_model == "paraformer-realtime-v2"
    assert settings.aliyun_asr_sample_rate == 16000
    assert settings.aliyun_asr_format == "pcm"


def test_settings_rejects_missing_provider_dependencies() -> None:
    cases = [
        {"asr_provider": "http", "asr_base_url": ""},
        {"asr_provider": "aliyun_ws", "aliyun_asr_api_key": ""},
        {
            "speaker_diarization_provider": "http",
            "speaker_diarization_base_url": "",
        },
        {"aigc_detector_provider": "http", "aigc_detector_base_url": ""},
        {"rate_limit_backend": "redis", "redis_url": ""},
        {"offline_task_backend": "redis_stream", "redis_url": ""},
        {"offline_task_backend": "celery", "celery_broker_url": ""},
        {"offline_task_backend": "celery", "celery_result_backend": ""},
        {
            "object_storage_endpoint": "http://minio:9000",
            "object_storage_access_key": "access",
            "object_storage_secret_key": "",
        },
    ]

    for kwargs in cases:
        with pytest.raises(ValidationError):
            Settings(**kwargs)
