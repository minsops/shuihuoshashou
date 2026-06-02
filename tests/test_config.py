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
        speaker_diarization_provider="http",
        aigc_detector_provider="http",
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
