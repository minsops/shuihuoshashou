# Shuihuo Killer

AI real-time interview probe assistant and anti-padding evaluation system.

This repository implements the engineering spec in `/Users/zhangyifan/Downloads/水货杀手_工程规格.md`
as a local-first Python MVP:

- Python 3.11+ with Pydantic v2 schemas.
- FastAPI services and gateway.
- Local SQLite persistence by default, with a PostgreSQL runtime adapter for deployment profiles.
- In-memory async events for local development, with explicit offline scoring events.
- Local task queue boundary for offline scoring, with optional Redis Streams task publication.
- Unified LLM client with mock mode and OpenAI-compatible HTTP mode for `mimo2.5pro`.
- Runtime LLM prompts are stored under `prompts/` and loaded by services instead of being embedded in code.
- ASR interface supports local stub mode and configurable HTTP cloud ASR adapters.
- ASR sessions deduplicate repeated final chunks, allow partial-to-final updates, learn local speaker clusters from known audio, and smooth short unknown-speaker gaps.
- End-to-end offline demo from JD + interview turns to probe, scoring, AIGC checks, and report.
- Interview turns are stored in both the interview context and a `qa_turns` table for auditability.
- WebSocket transcripts carry speaker/finality/timestamp metadata, support channel-based speaker mapping, and emit separate credibility events.
- Docker Compose declares the gateway plus PostgreSQL, Redis, and MinIO for local infrastructure.
- PostgreSQL core schema SQL is provided under `db/postgres` for compose initialization.
- Runtime database URL parsing supports SQLite and PostgreSQL targets.
- JD knowledge base indexes competency-specific probe patterns with deterministic embeddings and optional pgvector search.
- Report artifacts write local files by default and upload to S3-compatible storage when credentials are configured.
- Reports include structured scores, radar charts, highlights, AIGC checks, consistency flags,
  risk highlights, recommendations, and full interview transcripts.
- AIGC/template checks use a local answer-template corpus with character n-gram cosine similarity.

## Important Secret Handling

Do not commit API keys. Put them in `.env` or your shell:

```bash
export LLM_PROVIDER=openai_compatible
export LLM_MODEL=mimo-v2.5-pro
export LLM_BASE_URL="https://token-plan-cn.xiaomimimo.com/v1"
export LLM_API_PATH="/chat/completions"
export LLM_API_KEY="your-key"
export LLM_AUTH_HEADER="api-key"
export LLM_AUTH_SCHEME=""
export LLM_RESPONSE_CONTENT_PATH="choices.0.message.content"
export LLM_EXTRA_BODY_JSON=""
export LLM_MAX_RETRIES=1
```

If `LLM_PROVIDER=mock` or no API key is present, the system runs with deterministic local mock output.

The default values follow the MiMo OpenAI-compatible chat completions protocol. If the provider changes
the endpoint, auth header, response JSON shape, or retry policy, change only the `LLM_*` environment
variables above. LLM JSON parsing is validated with pydantic; failed HTTP/JSON/schema attempts retry
once by default before falling back to deterministic local behavior.

Set `GATEWAY_API_KEY` in deployed environments to require `X-API-Key: ...` or
`Authorization: Bearer ...` for `/api/*` and WebSocket traffic. It is empty by default for local demos.

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python scripts/run_offline_demo.py
uvicorn services.gateway.app:app --reload --port 8000
```

Open API docs at `http://127.0.0.1:8000/docs`.

Open the local demo UI at `http://127.0.0.1:8000/`.

## Run With Docker Compose

Docker Compose starts the gateway together with the infrastructure named in the engineering spec:
PostgreSQL, Redis, and S3-compatible MinIO.

```bash
docker compose up --build
```

The gateway uses SQLite in the default local profile. Set `DATABASE_URL` to a `postgresql://...`
URL and install `.[postgres]` to run the same repository calls against PostgreSQL.

PostgreSQL initializes from `db/postgres/001_core_schema.sql`, which declares the core jobs,
candidates, interviews, turns, probe-pattern embeddings, scores, AIGC results, reports, and consent tables from the spec.
The runtime adapter translates the local repository parameter style and upserts for PostgreSQL.
Set `JD_VECTOR_BACKEND=pgvector` on PostgreSQL deployments with pgvector installed to apply
`db/postgres/002_pgvector_probe_patterns.sql` and use `embedding_vector <=> query` nearest-neighbor
retrieval for probe-pattern search. The default `local` backend keeps Docker's plain PostgreSQL image
runnable.

Set `OBJECT_STORAGE_ENDPOINT`, `OBJECT_STORAGE_BUCKET`, `OBJECT_STORAGE_ACCESS_KEY`, and
`OBJECT_STORAGE_SECRET_KEY` to upload report HTML/PDF artifacts to S3-compatible storage such as
MinIO. The gateway still writes a local copy first for development and auditability.

Operational endpoints:

```bash
curl -s http://127.0.0.1:8000/metrics
RATE_LIMIT_ENABLED=true RATE_LIMIT_REQUESTS_PER_MINUTE=120 uvicorn services.gateway.app:app --port 8000
```

HTTP responses include `X-Request-ID` and W3C `traceparent`; clients may send the same headers to
correlate API calls, structured JSON logs, and Prometheus request metrics. Set
`OTEL_EXPORTER_OTLP_ENDPOINT` when deploying behind an OpenTelemetry collector; the current local
runtime emits trace-compatible IDs without requiring the SDK.

Check runtime configuration without exposing secrets:

```bash
python scripts/check_llm.py
python scripts/diagnose_llm_network.py
LLM_API_KEY=your-key python scripts/diagnose_llm_auth.py
curl -s http://127.0.0.1:8000/api/config/status
```

When `GATEWAY_API_KEY` is set, include `X-API-Key` or a bearer token on API requests. WebSocket
clients can pass the same key in headers or as `?api_key=...`.

If `check_llm.py` fails with a connection error, run `diagnose_llm_network.py`. It checks DNS, TCP,
and TLS for the configured `LLM_BASE_URL` without printing your API key.

For the default MiMo endpoint, `diagnose_llm_network.py` should report DNS, TCP, and TLS as OK before
API-key or request-format issues are diagnosed. If `check_llm.py` returns `HTTP 401 Invalid API Key`,
run `diagnose_llm_auth.py` to try common OpenAI-compatible auth header variants. If every variant
returns 401, generate a fresh key and set it through `LLM_API_KEY` without committing it.

## Main API

- `POST /api/jobs`
- `POST /api/candidates`
- `POST /api/consents`
- `POST /api/interviews`
- `POST /api/probe`
- `POST /api/offline/evaluate`
- `GET /api/config/status`
- `GET /metrics`
- `GET /api/jobs/{id}/probe-patterns?q=...`
- `POST /api/interviews/{id}/end`
- `GET /api/interviews/{id}/report`
- `GET /api/interviews/{id}/report.html`
- `GET /api/interviews/{id}/report.pdf`
- `WS /ws/interview/{id}`

WebSocket `audio_chunk` events may include `speaker`, `channel`/`audio_channel`/`track`,
`is_final`, `start_ms`, `end_ms`, and `confidence`. If `speaker` is omitted, channels listed in
`ASR_INTERVIEWER_CHANNELS` map to `interviewer`, and channels listed in `ASR_CANDIDATE_CHANNELS` map
to `candidate`. Only final candidate segments trigger a probe. Downstream events include
`transcript`, `probe`, `credibility`, optional `signal`, and `report`. Repeated final ASR chunks
with the same sequence are deduplicated and returned as `asr_warning` events instead of triggering
duplicate probe generation. When ASR returns `unknown` speaker, the session manager first tries to
resolve it from a previously observed local audio cluster, then falls back to short-gap continuity.

Set `ASR_PROVIDER=http`, `ASR_BASE_URL`, `ASR_API_PATH`, and `ASR_API_KEY` to forward audio chunks to
a cloud ASR endpoint. Response mapping is configurable with `ASR_TEXT_PATH`, `ASR_SPEAKER_PATH`,
`ASR_START_MS_PATH`, `ASR_END_MS_PATH`, `ASR_IS_FINAL_PATH`, and `ASR_CONFIDENCE_PATH`.

Set `SPEAKER_DIARIZATION_PROVIDER=http`, `SPEAKER_DIARIZATION_BASE_URL`,
`SPEAKER_DIARIZATION_API_PATH`, and `SPEAKER_DIARIZATION_API_KEY` to resolve unknown speakers through
a production voice-clustering service. The expected response speaker path defaults to
`SPEAKER_DIARIZATION_SPEAKER_PATH=speaker`.

## One-Shot Offline Evaluation

Use this endpoint for the first demo path: paste JD and interview Q&A, then receive the structured
report plus generated HTML/PDF paths.

In the local profile, `POST /api/interviews/{id}/end` publishes task events and runs the offline
pipeline synchronously so demos still return the report immediately. The persisted interview state
still follows the spec flow: `FINISHED -> SCORING -> REPORTED`.

The offline scoring task uses `OFFLINE_TASK_BACKEND=local` by default. Set
`OFFLINE_TASK_BACKEND=redis_stream` and install `.[redis]` to also publish task payloads to Redis
Streams under `{REDIS_STREAM_PREFIX}:tasks:{task_name}` while retaining synchronous local execution.
Set `OFFLINE_TASK_BACKEND=celery` and install `.[celery]` to publish the same
`interview.offline_scoring` task through Celery using `CELERY_BROKER_URL` and
`CELERY_RESULT_BACKEND`.

Set `OFFLINE_TASK_EXECUTION=async` to make `POST /api/interviews/{id}/end` return a queued task
instead of blocking for the report. The worker then consumes the Redis Stream and creates the report.
`POST /api/offline/evaluate` remains synchronous for demos and smoke tests.

Run a Redis Streams consumer for offline scoring tasks with:

```bash
OFFLINE_TASK_BACKEND=redis_stream python scripts/run_offline_worker.py
```

Use `--once` for a single poll cycle in deployment smoke tests.

Run a Celery worker for offline scoring tasks with:

```bash
OFFLINE_TASK_BACKEND=celery celery -A services.offline_worker.celery_tasks:celery_app worker --loglevel=info
```

```bash
curl -s http://127.0.0.1:8000/api/offline/evaluate \
  -H 'content-type: application/json' \
  -d '{
    "job_title": "AI 后端工程师",
    "jd_text": "Python FastAPI LLM 可靠性",
    "candidate_name": "候选人A",
    "turns": [
      {
        "question": "介绍一个核心项目",
        "answer": "我主要负责整体架构设计并推动项目落地最终取得显著提升",
        "answer_start_ms": 0,
        "answer_end_ms": 1000
      }
    ]
  }'
```

## Scope Notes

The real-time ASR and optional behavior signal modules are implemented behind interfaces with local
stub engines. Production speaker clustering can replace the local audio-cluster diarizer through the
HTTP diarization provider without changing the shared schemas.

Behavior signals are disabled by default. If an interview sets `signal_enabled=true`, the candidate
must first grant `behavior_signal` consent through `POST /api/consents`; otherwise the API returns 403.
Posting the same consent with `granted=false` revokes prior active behavior-signal consent and future
signal-enabled interviews are rejected until consent is granted again.
