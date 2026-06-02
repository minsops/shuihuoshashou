# Shuihuo Killer

AI real-time interview probe assistant and anti-padding evaluation system.

This repository implements the engineering spec in `/Users/zhangyifan/Downloads/水货杀手_工程规格.md`
as a local-first Python MVP:

- Python 3.11+ with Pydantic v2 schemas.
- FastAPI services and gateway.
- Local SQLite persistence by default, with a PostgreSQL runtime adapter for deployment profiles.
- In-memory async events for local development, with explicit offline scoring events.
- Local task queue boundary for offline scoring, with optional Redis Streams task publication.
- Interview state transitions reject post-report mutation and scoring before finish.
- Offline scoring rejects empty candidate-turn contexts so reports cannot contain unsupported scores.
- Unified LLM client with mock mode and OpenAI-compatible HTTP mode for `mimo2.5pro`.
- Runtime LLM prompts are stored under `prompts/` and loaded by services instead of being embedded in code.
- Probe responses are schema-limited to one to three suggestions, matching the realtime card contract.
- LLM-generated probe cards are sorted by priority and re-numbered before reaching realtime clients.
- ASR interface supports local stub mode and configurable HTTP cloud ASR adapters.
- ASR sessions deduplicate repeated final chunks, allow partial-to-final updates, learn local speaker clusters from known audio, and smooth short unknown-speaker gaps.
- Transcript segments reject blank session ids or transcript text before they enter orchestration.
- End-to-end offline demo from JD + interview turns to probe, scoring, AIGC checks, and report.
- Local demo UI includes both offline evaluation and realtime WebSocket probe panels.
- Interview turns are stored in both the interview context and a `qa_turns` table for auditability.
- Interview Q&A turns and scoring evidence excerpts reject blank text at the shared schema boundary.
- AI-probe Q&A turns require a non-blank `probe_target` so report transcripts keep the probe purpose.
- Job, candidate, competency, and probe-card contract fields reject blank required text.
- WebSocket transcripts carry speaker/finality/timestamp metadata, support channel-based speaker mapping, and emit separate credibility events.
- Docker Compose declares the gateway plus PostgreSQL, Redis, and MinIO for local infrastructure.
- PostgreSQL core schema SQL is provided under `db/postgres` for compose initialization.
- Runtime database URL parsing supports SQLite and PostgreSQL targets.
- Interview and consent creation validate candidate references so local SQLite behavior matches the
  PostgreSQL foreign-key contract.
- JD knowledge base indexes competency-specific probe patterns with deterministic embeddings and optional pgvector search.
- Probe-pattern retrieval scores reject infinite values so ranking remains deterministic.
- JD competency models are generated through the shared LLM JSON client with deterministic fallback
  and normalized to keep the required scoring dimensions and weights present.
- Competency and scoring weights reject NaN/inf values so Python-side weighted totals remain reproducible.
- Report artifacts write local files by default and upload to S3-compatible storage when credentials are configured.
- Reports include structured scores, radar charts, highlights, AIGC checks, consistency flags,
  de-duplicated risk highlights, recommendations, and full interview transcripts.
- Report building rejects mismatched score sessions, dimension names/weights, totals,
  recommendations, evidence turn ids, out-of-range evidence timestamps, non-transcript evidence
  excerpts, duplicate evidence references, unknown consistency-flag turn ids, incomplete/duplicate
  AIGC coverage, unknown AIGC turn ids, or unflagged AIGC results above configured thresholds.
- Report generation writes separate structured report JSON, HTML, PDF, and transcript JSON artifacts
  for audit and storage.
- Report artifact paths and URI entries reject blank values at the shared output boundary.
- PDF generation uses WeasyPrint when available and falls back to an auditable text PDF with score,
  recommendation, evidence, risk, AIGC, and transcript summary, including CJK text, when native
  rendering dependencies are missing.
- Interview context keeps an auditable fact-claim table for role, responsibility, technology, and
  metric statements used by consistency checks, including shared-context metric conflicts.
- Fact-claim responsibility, technology, and metric entries reject blank text before consistency checks.
- Consistency flags require two distinct transcript turns so risk highlights remain traceable.
- Scoring uses the shared LLM JSON client for structured dimension drafts and recomputes final
  totals in Python for auditability; deterministic risk signals cap affected dimensions even when
  an LLM draft returns higher scores, and deterministic risk notes are preserved in reports.
- Scoring evidence is normalized to real transcript spans and de-duplicated before reporting.
- Scoring risk notes reject blank entries at the shared schema boundary.
- Scoring schemas require at least one evidence reference per dimension.
- AIGC checks combine a local answer-template corpus with optional HTTP detector integration.
- AIGC detection requests reject empty turn batches and duplicate turn ids before scoring/reporting.
- AIGC matched template names reject blank text when a template hit is recorded.
- Interview turn writes reject duplicate turn ids so evidence, AIGC, and report references stay unambiguous.

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
export LLM_RATE_LIMIT_ENABLED=false
export LLM_RATE_LIMIT_REQUESTS_PER_MINUTE=60
```

If `LLM_PROVIDER=mock` or no API key is present, the system runs with deterministic local mock output.

The default values follow the MiMo OpenAI-compatible chat completions protocol. If the provider changes
the endpoint, auth header, response JSON shape, or retry policy, change only the `LLM_*` environment
variables above. LLM JSON parsing is validated with pydantic; failed HTTP/JSON/schema attempts retry
once by default before falling back to deterministic local behavior. The configured response path can
resolve to either a JSON string or an already-decoded JSON object.
Set `LLM_RATE_LIMIT_ENABLED=true` to cap provider calls per model before they leave the process;
limited calls fall back locally unless diagnostic code asks for `raise_on_error`.

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

To run the optional Celery offline-scoring worker in the same compose stack:

```bash
OFFLINE_TASK_BACKEND=celery OFFLINE_TASK_EXECUTION=async docker compose --profile worker up --build
```

The compose gateway uses PostgreSQL by default through
`postgresql://shuihuo:shuihuo_local@postgres:5432/shuihuo_killer`, while the non-Docker local
profile still defaults to SQLite for quick demos.
The compose stack forwards the runtime integration knobs from `.env.example` into the relevant
services, including gateway auth, LLM retry/rate-limit settings, ASR response mappings, probe
trigger thresholds, external AIGC detector settings, Redis rate-limit settings, and Celery worker
AIGC/LLM settings.

PostgreSQL initializes from `db/postgres/001_core_schema.sql`, which declares the core jobs,
candidates, interviews, turns, probe-pattern embeddings, scores, AIGC results, reports, and consent
tables from the spec. The schema also applies CHECK constraints for interview states, turn
timestamps, question sources, score ranges, AIGC probability/similarity ranges, recommendations, and
behavior-signal consent type. Interview status values must match their `started_at`/`ended_at`
timestamps in shared models, the PostgreSQL core schema, and freshly initialized local SQLite
schemas, and SQLite interview rows keep `signal_enabled` as a 0/1 boolean. Fresh local SQLite
`qa_turns` tables also enforce the same turn-index, source, text, time
range, and probe-target checks used by the PostgreSQL audit table. Fresh SQLite `aigc_results`
tables enforce the same probability, template-similarity, matched-template, and flag bounds. Fresh
SQLite `scores` tables enforce non-empty dimension arrays, score bounds, risk-note arrays, and
recommendation values. Fresh SQLite job, candidate, and probe-pattern tables reject blank text where
the PostgreSQL core schema does. Fresh SQLite consent tables enforce consent type, boolean grant
values, and revocation timestamps. Fresh SQLite report tables reject blank HTML artifacts. Fresh
SQLite JSON-backed columns reject invalid JSON text and wrong JSON shapes for object/vector fields.
The runtime adapter translates the local repository parameter style and upserts for PostgreSQL.
The default compose stack mounts only the core schema into Postgres initialization so Docker's plain
PostgreSQL image remains runnable. Set `JD_VECTOR_BACKEND=pgvector` on PostgreSQL deployments with
pgvector installed to apply `db/postgres/002_pgvector_probe_patterns.sql` at application startup and
use `embedding_vector <=> query` nearest-neighbor retrieval for probe-pattern search.

Set `OBJECT_STORAGE_ENDPOINT`, `OBJECT_STORAGE_BUCKET`, `OBJECT_STORAGE_ACCESS_KEY`, and
`OBJECT_STORAGE_SECRET_KEY` to upload report HTML/PDF artifacts to S3-compatible storage such as
MinIO. Transcript JSON artifacts are uploaded through the same store. The gateway still writes a
local copy first for development and auditability.

Operational endpoints:

```bash
curl -s http://127.0.0.1:8000/metrics
RATE_LIMIT_ENABLED=true RATE_LIMIT_REQUESTS_PER_MINUTE=120 uvicorn services.gateway.app:app --port 8000
```

Set `RATE_LIMIT_BACKEND=redis` and install `.[redis]` to share gateway rate-limit counters across
processes through Redis. Local development uses the in-memory backend by default.

HTTP responses include `X-Request-ID` and W3C `traceparent`; clients may send the same headers to
correlate API calls, structlog-rendered JSON logs, and Prometheus request metrics. Set
`OTEL_EXPORTER_OTLP_ENDPOINT` and install the `.[otel]` extra when deploying behind an
OpenTelemetry collector; Docker images include that extra and instrument the FastAPI gateway when
the endpoint is configured. The metrics endpoint also exposes domain/task event counters such as
`task.enqueued`, `task.completed`, `task.failed`, `task.worker_failed`, `interview.finished`, and
`interview.reported` for the offline scoring path.

Check runtime configuration without exposing secrets:

```bash
python scripts/check_llm.py
python scripts/diagnose_llm_network.py
LLM_API_KEY=your-key python scripts/diagnose_llm_auth.py
curl -s http://127.0.0.1:8000/api/config/status
```

`/api/config/status` reports non-secret provider paths, response mapping paths, timeout values,
OTLP exporter presence, and whether secrets are configured, but never returns API keys. Database
URLs are returned with passwords redacted.
Provider/backend selectors, numeric thresholds, and required companion URLs are validated when
settings load, so unsupported or incomplete values fail fast during startup or smoke tests.

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
- `POST /api/aigc/detect`
- `POST /api/scoring/score`
- `POST /api/report/build`
- `POST /api/offline/evaluate`
- `GET /api/config/status`
- `GET /metrics`
- `GET /api/jobs/{id}/probe-patterns?q=...`
- `POST /api/interviews/{id}/end`
- `GET /api/interviews/{id}/report`
- `GET /api/interviews/{id}/report.html`
- `GET /api/interviews/{id}/report.json`
- `GET /api/interviews/{id}/report.pdf`
- `GET /api/interviews/{id}/report.transcript.json`
- `WS /ws/interview/{id}`

WebSocket `audio_chunk` events may include `speaker`, `channel`/`audio_channel`/`track`,
`is_final`, `start_ms`, `end_ms`, and `confidence`. If `speaker` is omitted, channels listed in
`ASR_INTERVIEWER_CHANNELS` map to `interviewer`, and channels listed in `ASR_CANDIDATE_CHANNELS` map
to `candidate`. Only final candidate segments trigger a probe. Downstream events include
`transcript`, `probe`, `credibility`, optional `signal`, `report`, and async-mode `task_queued`.
Repeated final ASR chunks
with the same sequence are deduplicated and returned as `asr_warning` events instead of triggering
duplicate probe generation. When ASR returns `unknown` speaker, the session manager first tries to
resolve it from a previously observed local audio cluster, then falls back to short-gap continuity.
If an `audio_chunk` includes a `session_id`, it must match the WebSocket interview id; mismatches are
returned as `asr_warning` events and skipped.
Invalid or empty `audio_chunk.audio` payloads are rejected with `asr_warning` instead of being
converted into placeholder transcripts. Blank `text_turn.answer` values are rejected with an
`error` event for the same reason. Binary frames, invalid JSON text frames, and non-object
WebSocket JSON payloads are rejected with an `error` event while keeping the session open for later
valid events. Premature `end` events that fail state guards also return an `error` event without
closing the session. If audio metadata is provided, the gateway accepts only PCM/Opus-style
formats, `sample_rate_hz=16000`, and `channels=1`; unsupported values are rejected before reaching
ASR.
Tune `PROBE_MIN_ANSWER_CHARS` and `PROBE_MIN_INTERVAL_MS` to control when candidate final segments
are eligible for probe generation. `PROBE_REQUIRE_TOPIC_MATCH` and `PROBE_TOPIC_KEYWORDS` keep
automatic probes focused on drill-down topics such as projects, technical decisions, metrics, and
incidents. Send a WebSocket `manual_probe` event with an `answer` to model the interviewer clicking
"ask a probe now"; manual probes bypass the automatic length, topic, and interval gates.

Set `ASR_PROVIDER=http`, `ASR_BASE_URL`, `ASR_API_PATH`, and `ASR_API_KEY` to forward audio chunks to
a cloud ASR endpoint. Response mapping is configurable with `ASR_TEXT_PATH`, `ASR_SPEAKER_PATH`,
`ASR_START_MS_PATH`, `ASR_END_MS_PATH`, `ASR_IS_FINAL_PATH`, and `ASR_CONFIDENCE_PATH`. Finality
strings such as `partial`, `interim`, and `provisional` are treated as non-final so provisional ASR
output does not trigger probes. ASR timestamps are normalized to non-negative millisecond ranges
with `end_ms >= start_ms`; the shared transcript, Q&A, and scoring evidence schemas reject invalid
timestamp ranges at API boundaries as well. Interview contexts and records also reject `ended_at`
values earlier than their `started_at` timestamps.
If the ASR provider fails or returns an invalid transcript for one frame, the WebSocket emits
`asr_warning` with `reason=asr_transcription_failed` and keeps the interview session open for later
frames.

Set `SPEAKER_DIARIZATION_PROVIDER=http`, `SPEAKER_DIARIZATION_BASE_URL`,
`SPEAKER_DIARIZATION_API_PATH`, and `SPEAKER_DIARIZATION_API_KEY` to resolve unknown speakers through
a production voice-clustering service. The expected response speaker path defaults to
`SPEAKER_DIARIZATION_SPEAKER_PATH=speaker`. Docker Compose forwards the same
`SPEAKER_DIARIZATION_*` variables to the gateway, and the gateway refreshes the ASR session
diarizer from runtime settings during startup.

Set `AIGC_DETECTOR_PROVIDER=http`, `AIGC_DETECTOR_BASE_URL`, `AIGC_DETECTOR_API_PATH`, and
`AIGC_DETECTOR_API_KEY` to send each answer to an external AI-text detector. The local template
similarity result and configured probability threshold remain part of the final flag decision; HTTP
failures fall back to the deterministic local detector.

## One-Shot Offline Evaluation

Use this endpoint for the first demo path: paste JD and interview Q&A, then receive the structured
report plus generated HTML/PDF paths.

In the local profile, `POST /api/interviews/{id}/end` publishes task events and runs the offline
pipeline synchronously so demos still return the report immediately. The persisted interview state
still follows the spec flow: the first turn write starts a created interview, then end advances
`IN_PROGRESS -> FINISHED -> SCORING -> REPORTED`.

The offline scoring task uses `OFFLINE_TASK_BACKEND=local` by default. Set
`OFFLINE_TASK_BACKEND=redis_stream` and install `.[redis]` to also publish task payloads to Redis
Streams under `{REDIS_STREAM_PREFIX}:tasks:{task_name}` while retaining synchronous local execution.
Set `OFFLINE_TASK_BACKEND=celery` and install `.[celery]` to publish the same
`interview.offline_scoring` task through Celery using `CELERY_BROKER_URL` and
`CELERY_RESULT_BACKEND`. The publisher and worker both use `CELERY_TASK_QUEUE`
(`shuihuo-offline` by default), so custom deployments must set the same queue name on both sides.

Set `OFFLINE_TASK_EXECUTION=async` to make `POST /api/interviews/{id}/end` return a queued task
instead of blocking for the report. Once queued, the interview advances to `SCORING` so repeated
end requests cannot enqueue duplicate offline scoring tasks. The worker then consumes the Redis
Stream and creates the report. `POST /api/offline/evaluate` remains synchronous for demos and smoke
tests.

Run a Redis Streams consumer for offline scoring tasks with:

```bash
OFFLINE_TASK_BACKEND=redis_stream python scripts/run_offline_worker.py
```

Use `--once` for a single poll cycle in deployment smoke tests.
Redis Streams workers emit `task.worker_failed` for handler errors or malformed payloads and leave
failed messages unacknowledged for retry or manual inspection.

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

Behavior signals are disabled by default. `SIGNAL_ENABLED=true` must be set by an administrator
before any interview can request `signal_enabled=true`, and the candidate must also grant
`behavior_signal` consent through `POST /api/consents`; otherwise the API returns 403. Posting the
same consent with `granted=false` revokes prior active behavior-signal consent and future
signal-enabled interviews are rejected until both gates are satisfied again. Consent checks use
database-parameterized booleans so the same flow works on SQLite and PostgreSQL. Shared consent
records also reject revoked timestamps earlier than the grant timestamp before persistence. The
realtime WebSocket path also re-checks active consent before emitting each optional `signal` event,
so revoked consent suppresses further behavior-signal hints.
