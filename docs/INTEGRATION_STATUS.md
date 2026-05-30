# Integration Status

## Current State

The local implementation is complete as a runnable MVP:

- FastAPI gateway and API docs.
- Local demo UI at `/`.
- One-shot offline evaluation at `/api/offline/evaluate`.
- WebSocket real-time text/audio-stub probe flow with speaker/finality/timestamp metadata.
- Separate `credibility` WebSocket event after probe generation.
- Pydantic v2 shared schemas.
- SQLite local persistence.
- PostgreSQL core schema SQL for compose initialization.
- Database URL dialect detection and runtime connection support for SQLite and PostgreSQL targets.
- Separate `qa_turns` persistence for auditable answer evidence.
- JD knowledge base local retrieval for competency-specific probe patterns.
- Probe generation, scoring, AIGC/template checks, consistency checks, HTML/PDF report generation.
- AIGC template checks use a local corpus and character n-gram cosine similarity.
- Structured JSON and HTML reports include the full interview transcript.
- Report artifacts support local `file://` storage and SigV4 uploads to S3-compatible storage.
- Local offline scoring task flow with `FINISHED -> SCORING -> REPORTED` state transitions.
- Local task queue boundary for offline scoring with enqueue/completion/failure events.
- In-memory event bus topics for `qa_turn.created`, `interview.finished`,
  `task.enqueued`, `interview.scoring_started`, `interview.reported`, and `task.completed`.
- Behavior signal module with explicit candidate consent gate.
- Candidate behavior-signal consent can be revoked and blocks future signal-enabled interviews.
- Configurable LLM client with mock mode and OpenAI-compatible HTTP mode.
- Safe runtime config and LLM smoke-test scripts.
- Prometheus-style `/metrics` endpoint for local HTTP request counters and duration sums.
- Optional in-memory per-client rate limit gate for the gateway.
- Docker Compose entrypoint for gateway, PostgreSQL, Redis, and MinIO.

## MiMo Configuration

Default MiMo settings are configured through environment variables:

```bash
LLM_PROVIDER=openai_compatible
LLM_MODEL=mimo-v2.5-pro
LLM_BASE_URL=https://token-plan-cn.xiaomimimo.com/v1
LLM_API_PATH=/chat/completions
LLM_AUTH_HEADER=api-key
LLM_AUTH_SCHEME=
LLM_RESPONSE_CONTENT_PATH=choices.0.message.content
```

The API key must be supplied locally through `.env` or the shell. Do not commit it.

The previously documented `https://api.mimo-v2.com/v1` endpoint did not resolve from this machine.
The current default is the user-provided OpenAI-compatible endpoint above.

## Live Smoke Status

Real MiMo API smoke testing reaches the current endpoint and succeeds when a valid API key is supplied
through the shell or local `.env`. Keep the key out of git.

Recent verified command shape:

```bash
LLM_PROVIDER=openai_compatible LLM_API_KEY=<valid-key> python scripts/check_llm.py
```

Expected result:

```text
LLM smoke test ok
```

If a future key fails, run `python scripts/diagnose_llm_network.py` first to separate network issues
from auth or request-format issues.

## Verification

Local verification command:

```bash
python -m pytest -q
ruff check .
python scripts/check_llm.py
```

Expected current local result in mock mode:

```text
pytest: passing
ruff: passing
LLM smoke test ok
```

## Remaining Production Gaps

- Add pgvector-backed semantic retrieval for JD/probe-pattern search.
- Replace the local task queue/event bus with Redis Streams, RabbitMQ, Kafka, or Celery workers.
- Plug a real streaming ASR/diarization engine behind the existing `ASREngine` interface.
