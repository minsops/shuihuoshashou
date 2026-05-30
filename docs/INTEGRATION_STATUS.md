# Integration Status

## Current State

The local implementation is complete as a runnable MVP:

- FastAPI gateway and API docs.
- Optional gateway API-key authentication for `/api/*` and WebSocket traffic.
- Local demo UI at `/`.
- One-shot offline evaluation at `/api/offline/evaluate`.
- WebSocket real-time text/audio-stub probe flow with speaker/finality/timestamp metadata and channel-based speaker mapping.
- Configurable real-time probe trigger rules for candidate answer length, drill-down topic matching,
  minimum interval, and interviewer-initiated manual probes.
- Configurable HTTP cloud ASR adapter behind the `ASREngine` interface.
- ASR session manager supports partial-to-final chunk updates, repeated-final deduplication,
  stale-final rejection, local audio-cluster speaker resolution, and conservative short-gap
  speaker continuity smoothing.
- Speaker diarization is pluggable: local deterministic audio clusters for development, or an
  HTTP provider for production voice embedding / cloud diarization services.
- Separate `credibility` WebSocket event after probe generation.
- Pydantic v2 shared schemas.
- SQLite local persistence.
- PostgreSQL core schema SQL for compose initialization.
- Database URL dialect detection and runtime connection support for SQLite and PostgreSQL targets.
- Separate `qa_turns` persistence for auditable answer evidence.
- JD knowledge base indexes competency-specific probe patterns with deterministic embeddings and optional pgvector nearest-neighbor search.
- Probe generation, scoring, AIGC/template checks, consistency checks, HTML/PDF report generation.
- Runtime probe prompts are kept under `prompts/` and loaded by services instead of being embedded in code.
- AIGC template checks use a local corpus and character n-gram cosine similarity.
- Structured JSON and HTML/PDF reports include highlights, radar charts, AIGC risk highlights,
  recommendation, and the full interview transcript.
- Report artifacts support local `file://` storage and SigV4 uploads to S3-compatible storage.
- Local offline scoring task flow with `FINISHED -> SCORING -> REPORTED` state transitions.
- Local task queue boundary for offline scoring with Redis Streams task publication and worker consumption.
- Optional Celery task publication and worker registration for `interview.offline_scoring`.
- Configurable async end-interview mode returns queued task metadata while workers generate reports.
- In-memory event bus topics for `qa_turn.created`, `interview.finished`,
  `task.enqueued`, `interview.scoring_started`, `interview.reported`, and `task.completed`.
- Behavior signal module with explicit candidate consent gate.
- Candidate behavior-signal consent can be revoked and blocks future signal-enabled interviews.
- Configurable LLM client with mock mode and OpenAI-compatible HTTP mode.
- LLM JSON responses are pydantic-validated with configurable retry before deterministic fallback.
- Safe runtime config and LLM smoke-test scripts.
- Prometheus-style `/metrics` endpoint for local HTTP request counters and duration sums.
- Structured JSON request logs with propagated `X-Request-ID` correlation IDs.
- W3C `traceparent` propagation with trace/span IDs included in request logs.
- Optional per-client gateway rate limit gate with local and Redis-backed counter modes.
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

- Use a real production diarization endpoint or pyannote-backed service behind the HTTP provider in
  deployed environments.
