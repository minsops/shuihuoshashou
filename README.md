# Shuihuo Killer

AI real-time interview probe assistant and anti-padding evaluation system.

This repository implements the engineering spec in `/Users/zhangyifan/Downloads/水货杀手_工程规格.md`
as a local-first Python MVP:

- Python 3.11+ with Pydantic v2 schemas.
- FastAPI services and gateway.
- Local SQLite persistence by default, with a PostgreSQL runtime adapter for deployment profiles.
- In-memory async events instead of Redis/MQ for local development, with explicit offline scoring events.
- Local task queue boundary for offline scoring, ready to swap for Celery/Redis workers.
- Unified LLM client with mock mode and OpenAI-compatible HTTP mode for `mimo2.5pro`.
- End-to-end offline demo from JD + interview turns to probe, scoring, AIGC checks, and report.
- Interview turns are stored in both the interview context and a `qa_turns` table for auditability.
- WebSocket transcripts carry speaker/finality/timestamp metadata and emit separate credibility events.
- Docker Compose declares the gateway plus PostgreSQL, Redis, and MinIO for local infrastructure.
- PostgreSQL core schema SQL is provided under `db/postgres` for compose initialization.
- Runtime database URL parsing supports SQLite and PostgreSQL targets.
- JD knowledge base exposes local lexical retrieval for competency-specific probe patterns.
- Report artifacts write local files by default and upload to S3-compatible storage when credentials are configured.
- Reports include structured scores, AIGC checks, consistency flags, and full interview transcripts.
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
```

If `LLM_PROVIDER=mock` or no API key is present, the system runs with deterministic local mock output.

The default values follow the MiMo OpenAI-compatible chat completions protocol. If the provider changes
the endpoint, auth header, or response JSON shape, change only the `LLM_*` environment variables above.

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
candidates, interviews, turns, scores, AIGC results, reports, and consent tables from the spec.
The runtime adapter translates the local repository parameter style and upserts for PostgreSQL.

Set `OBJECT_STORAGE_ENDPOINT`, `OBJECT_STORAGE_BUCKET`, `OBJECT_STORAGE_ACCESS_KEY`, and
`OBJECT_STORAGE_SECRET_KEY` to upload report HTML/PDF artifacts to S3-compatible storage such as
MinIO. The gateway still writes a local copy first for development and auditability.

Operational endpoints:

```bash
curl -s http://127.0.0.1:8000/metrics
RATE_LIMIT_ENABLED=true RATE_LIMIT_REQUESTS_PER_MINUTE=120 uvicorn services.gateway.app:app --port 8000
```

Check runtime configuration without exposing secrets:

```bash
python scripts/check_llm.py
python scripts/diagnose_llm_network.py
LLM_API_KEY=your-key python scripts/diagnose_llm_auth.py
curl -s http://127.0.0.1:8000/api/config/status
```

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

WebSocket `audio_chunk` events may include `speaker`, `is_final`, `start_ms`, `end_ms`, and
`confidence`. Only final candidate segments trigger a probe. Downstream events include
`transcript`, `probe`, `credibility`, optional `signal`, and `report`.

## One-Shot Offline Evaluation

Use this endpoint for the first demo path: paste JD and interview Q&A, then receive the structured
report plus generated HTML/PDF paths.

In the local profile, `POST /api/interviews/{id}/end` publishes local task events and runs the
offline pipeline synchronously so demos still return the report immediately. The persisted interview
state still follows the spec flow: `FINISHED -> SCORING -> REPORTED`.

The offline scoring task uses `OFFLINE_TASK_BACKEND=local` by default. It records task enqueue,
completion, and failure events so the Celery/Redis worker boundary is explicit.

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
stub engines. Production ASR, diarization, object storage, vector DB, Redis Streams, and Celery can be
plugged in without changing the shared schemas.

Behavior signals are disabled by default. If an interview sets `signal_enabled=true`, the candidate
must first grant `behavior_signal` consent through `POST /api/consents`; otherwise the API returns 403.
Posting the same consent with `granted=false` revokes prior active behavior-signal consent and future
signal-enabled interviews are rejected until consent is granted again.
