from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_docker_compose_declares_required_infrastructure() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert compose.startswith("name: shuihuo-killer")
    for service in ["gateway:", "offline-worker:", "postgres:", "redis:", "minio:"]:
        assert service in compose
    assert "postgres:16-alpine" in compose
    assert "redis:7-alpine" in compose
    assert "minio/minio" in compose
    assert "GATEWAY_API_KEY: ${GATEWAY_API_KEY:-}" in compose
    assert "DATABASE_URL: postgresql://shuihuo:shuihuo_local@postgres:5432/shuihuo_killer" in compose
    assert "REDIS_URL: redis://redis:6379/0" in compose
    assert "OBJECT_STORAGE_ENDPOINT: http://minio:9000" in compose
    assert "OTEL_EXPORTER_OTLP_ENDPOINT: ${OTEL_EXPORTER_OTLP_ENDPOINT:-}" in compose
    assert "OTEL_SERVICE_NAME: ${OTEL_SERVICE_NAME:-shuihuo-killer-gateway}" in compose
    assert "SPEAKER_DIARIZATION_PROVIDER: ${SPEAKER_DIARIZATION_PROVIDER:-local}" in compose
    assert "SPEAKER_DIARIZATION_BASE_URL: ${SPEAKER_DIARIZATION_BASE_URL:-}" in compose
    assert "SPEAKER_DIARIZATION_API_PATH: ${SPEAKER_DIARIZATION_API_PATH:-/diarize}" in compose
    assert "SPEAKER_DIARIZATION_API_KEY: ${SPEAKER_DIARIZATION_API_KEY:-}" in compose
    assert "SPEAKER_DIARIZATION_SPEAKER_PATH: ${SPEAKER_DIARIZATION_SPEAKER_PATH:-speaker}" in compose
    for key in [
        "LLM_TIMEOUT_SECONDS: ${LLM_TIMEOUT_SECONDS:-30}",
        "LLM_MAX_RETRIES: ${LLM_MAX_RETRIES:-1}",
        "LLM_RATE_LIMIT_ENABLED: ${LLM_RATE_LIMIT_ENABLED:-false}",
        "LLM_RATE_LIMIT_REQUESTS_PER_MINUTE: ${LLM_RATE_LIMIT_REQUESTS_PER_MINUTE:-60}",
        "AIGC_DETECTOR_PROVIDER: ${AIGC_DETECTOR_PROVIDER:-local}",
        "AIGC_DETECTOR_BASE_URL: ${AIGC_DETECTOR_BASE_URL:-}",
        "AIGC_DETECTOR_API_PATH: ${AIGC_DETECTOR_API_PATH:-/detect}",
        "AIGC_DETECTOR_API_KEY: ${AIGC_DETECTOR_API_KEY:-}",
        "AIGC_AI_PROB_THRESHOLD: ${AIGC_AI_PROB_THRESHOLD:-0.65}",
        "AIGC_TEMPLATE_SIMILARITY_THRESHOLD: ${AIGC_TEMPLATE_SIMILARITY_THRESHOLD:-0.45}",
    ]:
        assert compose.count(key) == 2
    for key in [
        "ASR_TEXT_PATH: ${ASR_TEXT_PATH:-text}",
        "ASR_START_MS_PATH: ${ASR_START_MS_PATH:-start_ms}",
        "ASR_CONFIDENCE_PATH: ${ASR_CONFIDENCE_PATH:-confidence}",
        "PROBE_MIN_ANSWER_CHARS: ${PROBE_MIN_ANSWER_CHARS:-20}",
        "PROBE_REQUIRE_TOPIC_MATCH: ${PROBE_REQUIRE_TOPIC_MATCH:-true}",
        "RATE_LIMIT_BACKEND: ${RATE_LIMIT_BACKEND:-local}",
        "REDIS_RATE_LIMIT_PREFIX: ${REDIS_RATE_LIMIT_PREFIX:-shuihuo:rate_limit}",
        "ALIYUN_AK_ID: ${ALIYUN_AK_ID:-}",
        "ALIYUN_AK_SECRET: ${ALIYUN_AK_SECRET:-}",
        "ALIYUN_NLS_TOKEN_ENDPOINT: ${ALIYUN_NLS_TOKEN_ENDPOINT:-https://nls-meta.cn-shanghai.aliyuncs.com/}",
        "ALIYUN_NLS_TOKEN_REGION: ${ALIYUN_NLS_TOKEN_REGION:-cn-shanghai}",
    ]:
        assert key in compose
    assert 'profiles: ["worker"]' in compose
    assert "services.offline_worker.celery_tasks:celery_app" in compose
    assert "CELERY_BROKER_URL: redis://redis:6379/1" in compose
    assert "CELERY_RESULT_BACKEND: redis://redis:6379/2" in compose
    assert compose.count("CELERY_TASK_QUEUE: ${CELERY_TASK_QUEUE:-shuihuo-offline}") == 2
    assert (
        "./db/postgres/001_core_schema.sql:/docker-entrypoint-initdb.d/001_core_schema.sql:ro"
        in compose
    )
    assert "./db/postgres:/docker-entrypoint-initdb.d:ro" not in compose
    assert "002_pgvector_probe_patterns.sql:/docker-entrypoint-initdb.d" not in compose
    assert "healthcheck:" in compose


def test_dockerfile_packages_gateway_app() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "FROM python:3.11-slim" in dockerfile
    assert 'pip install --no-cache-dir -e ".[postgres,redis,celery,otel]"' in dockerfile
    assert "uvicorn" in dockerfile
    assert "services.gateway.app:app" in dockerfile


def test_pyproject_declares_optional_worker_dependencies() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'redis = ["redis>=5.0.0"]' in pyproject
    assert 'celery = ["celery[redis]>=5.3.0"]' in pyproject
    assert "opentelemetry-sdk>=1.26.0" in pyproject
    assert "opentelemetry-exporter-otlp-proto-http>=1.26.0" in pyproject
    assert "opentelemetry-instrumentation-fastapi>=0.47b0" in pyproject


def test_readme_documents_current_default_llm_model() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "`mimo-v2.5-pro`" in readme
    assert "`mimo2.5pro`" not in readme


def test_readme_documents_local_ocr_and_nls_asr_setup() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "`pip install -e '.[ocr]'`" in readme
    assert "`ASR_PROVIDER=aliyun_nls_ws`" in readme
    assert "阿里云智能语音交互 NLS WebSocket ASR" in readme


def test_readme_distinguishes_real_llm_smoke_from_mock_mode() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "LLM smoke test ok." in readme
    assert "LLM mock mode ok. No real model endpoint was called." in readme
    assert "只有 `LLM_PROVIDER=openai_compatible`" in readme
    assert "`LLM_EXTRA_BODY_JSON` 只能填写 JSON object" in readme
    assert "`LLM_RESPONSE_CONTENT_PATH` 必须是非空点分路径" in readme
    assert "响应字段映射 path 同样必须是非空点分路径" in readme
    assert "对应的 auth header 不能为空" in readme


def test_readme_explains_demo_gateway_key_input() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "`Gateway Key` 输入框" in readme
    assert "只保存在当前浏览器会话中" in readme
    assert "页面会自动带到 API、WebSocket 和报告下载链接" in readme


def test_readme_explains_empty_asr_smoke_result_semantics() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "`--allow-empty-result`" in readme
    assert "session completed, but no transcript text was verified" in readme
    assert "只验证了 WebSocket 会话完成，没有验证识别文本有效" in readme


def test_readme_explains_aliyun_nls_auto_token_mode() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "固定 Token 优先级最高" in readme
    assert "gateway 会在连接 NLS WebSocket 前自动创建 Token" in readme
    assert "ALIYUN_AK_ID=your-access-key-id" in readme


def test_env_example_lists_runtime_integration_knobs() -> None:
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")

    for key in [
        "ASR_PROVIDER=",
        "ASR_BASE_URL=",
        "GATEWAY_API_KEY=",
        "LLM_MAX_RETRIES=",
        "LLM_RATE_LIMIT_ENABLED=",
        "LLM_RATE_LIMIT_REQUESTS_PER_MINUTE=",
        "OTEL_EXPORTER_OTLP_ENDPOINT=",
        "OTEL_SERVICE_NAME=",
        "SPEAKER_DIARIZATION_PROVIDER=",
        "SPEAKER_DIARIZATION_BASE_URL=",
        "OFFLINE_TASK_EXECUTION=",
        "CELERY_BROKER_URL=",
        "CELERY_RESULT_BACKEND=",
        "CELERY_TASK_QUEUE=",
        "REDIS_STREAM_PREFIX=",
        "JD_VECTOR_BACKEND=",
        "OBJECT_STORAGE_ACCESS_KEY=",
        "REPORT_DIR=",
        "RATE_LIMIT_BACKEND=",
        "REDIS_RATE_LIMIT_PREFIX=",
        "PROBE_MIN_ANSWER_CHARS=",
        "PROBE_MIN_INTERVAL_MS=",
        "PROBE_REQUIRE_TOPIC_MATCH=",
        "PROBE_TOPIC_KEYWORDS=",
        "AIGC_DETECTOR_PROVIDER=",
        "AIGC_DETECTOR_BASE_URL=",
        "AIGC_AI_PROB_THRESHOLD=",
        "AIGC_TEMPLATE_SIMILARITY_THRESHOLD=",
        "ALIYUN_AK_ID=",
        "ALIYUN_AK_SECRET=",
        "ALIYUN_NLS_TOKEN_ENDPOINT=",
        "ALIYUN_NLS_TOKEN_REGION=",
    ]:
        assert key in env_example


def test_env_example_includes_complete_deepseek_configuration_block() -> None:
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")

    for line in [
        "# LLM_PROVIDER=openai_compatible",
        "# LLM_MODEL=deepseek-v4-pro",
        "# LLM_BASE_URL=https://api.deepseek.com",
        "# LLM_API_PATH=/chat/completions",
        "# LLM_API_KEY=your-deepseek-key",
        "# LLM_AUTH_HEADER=Authorization",
        "# LLM_AUTH_SCHEME=Bearer",
        "# LLM_RESPONSE_CONTENT_PATH=choices.0.message.content",
    ]:
        assert line in env_example


def test_postgres_schema_matches_core_spec_tables() -> None:
    schema = (ROOT / "db" / "postgres" / "001_core_schema.sql").read_text(encoding="utf-8")

    required_tables = [
        "jobs",
        "candidates",
        "interviews",
        "qa_turns",
        "scores",
        "aigc_results",
        "reports",
        "consents",
    ]
    for table in required_tables:
        assert f"CREATE TABLE IF NOT EXISTS {table}" in schema
    for column in [
        "competency_model JSONB NOT NULL",
        "context JSONB NOT NULL",
        "dimensions JSONB NOT NULL",
        "ai_generated_prob REAL NOT NULL",
        "template_similarity REAL NOT NULL",
        "signal_enabled BOOLEAN NOT NULL DEFAULT false",
    ]:
        assert column in schema
    assert "REFERENCES jobs(id)" in schema
    assert "REFERENCES candidates(id)" in schema
    assert "REFERENCES interviews(id)" in schema


def test_postgres_schema_enforces_core_contract_invariants() -> None:
    schema = (ROOT / "db" / "postgres" / "001_core_schema.sql").read_text(encoding="utf-8")

    for constraint in [
        "status IN ('CREATED', 'IN_PROGRESS', 'FINISHED', 'SCORING', 'REPORTED')",
        "ended_at IS NULL OR started_at IS NULL OR ended_at >= started_at",
        "turn_index >= 0",
        "question_source IN ('interviewer', 'ai_probe')",
        "answer_start_ms >= 0",
        "answer_end_ms >= answer_start_ms",
        "jsonb_typeof(dimensions) = 'array' AND jsonb_array_length(dimensions) > 0",
        "total_score >= 0 AND total_score <= 100",
        "jsonb_typeof(risk_notes) = 'array'",
        "recommendation IN ('strong_yes', 'yes', 'hold', 'no')",
        "ai_generated_prob >= 0 AND ai_generated_prob <= 1",
        "template_similarity >= 0 AND template_similarity <= 1",
        "consent_type IN ('behavior_signal')",
        "revoked_at IS NULL OR revoked_at >= granted_at",
        "status = 'CREATED' AND started_at IS NULL AND ended_at IS NULL",
        "status = 'IN_PROGRESS' AND started_at IS NOT NULL AND ended_at IS NULL",
        "status IN ('FINISHED', 'SCORING', 'REPORTED')",
    ]:
        assert constraint in schema
