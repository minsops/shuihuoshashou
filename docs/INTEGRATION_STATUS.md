# Integration Status

## Current State

The local implementation is complete as a runnable MVP:

- FastAPI gateway and API docs.
- Optional gateway API-key authentication for `/api/*` and WebSocket traffic.
- Local demo UI at `/` for offline evaluation and realtime WebSocket probe sessions.
- One-shot offline evaluation at `/api/offline/evaluate`.
- WebSocket real-time text/audio-stub probe flow with speaker/finality/timestamp metadata and channel-based speaker mapping.
- Invalid or empty WebSocket audio chunks are rejected with `asr_warning` before transcription.
- WebSocket audio chunks with mismatched `session_id` values are rejected before ASR processing.
- Configurable real-time probe trigger rules for candidate answer length, drill-down topic matching,
  minimum interval, and interviewer-initiated manual probes.
- Configurable HTTP cloud ASR adapter behind the `ASREngine` interface.
- HTTP ASR responses normalize string finality flags and clamp confidence values into the schema range.
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
- SQLite startup migrations backfill later realtime/report columns so older local demo databases stay usable.
- Separate `qa_turns` persistence for auditable answer evidence.
- JD knowledge base indexes competency-specific probe patterns with deterministic embeddings and optional pgvector nearest-neighbor search.
- JD competency generation uses the shared LLM JSON client with `prompts/competency_gen.md` and
  deterministic fallback.
- Probe generation, scoring, AIGC/template checks, consistency checks, HTML/PDF report generation.
- Internal HTTP-style contracts for standalone probe, AIGC detection, scoring, and report calls.
- Scoring uses the shared LLM JSON client for structured dimension drafts, then normalizes evidence
  and recomputes final totals in Python.
- Scoring evidence is re-anchored to persisted turns so fabricated excerpts or out-of-range
  timestamps from LLM drafts do not enter reports.
- Runtime probe prompts are kept under `prompts/` and loaded by services instead of being embedded in code.
- AIGC checks use a local corpus with character n-gram cosine similarity plus optional HTTP detector
  integration with deterministic local fallback.
- Interview context persists extracted fact claims for role, responsibility, technology, and metric
  statements, and consistency checks run against that fact table.
- Structured JSON and HTML/PDF reports include highlights, radar charts, AIGC risk highlights,
  recommendation, and the full interview transcript.
- Report artifacts support local `file://` storage and SigV4 uploads to S3-compatible storage for
  HTML, PDF, and transcript JSON outputs.
- Gateway exposes JSON, HTML, PDF, and transcript JSON report artifact endpoints.
- Local offline scoring task flow with `FINISHED -> SCORING -> REPORTED` state transitions.
- State guards prevent scoring before `FINISHED` and prevent turn edits or restarts after reporting.
- Local task queue boundary for offline scoring with Redis Streams task publication and worker consumption.
- Optional Celery task publication and worker registration for `interview.offline_scoring`.
- Docker Compose includes an optional `worker` profile for running the Celery offline-scoring worker.
- Configurable async end-interview mode returns queued task metadata while workers generate reports.
- Async end-interview advances queued interviews to `SCORING` and rejects duplicate queueing.
- In-memory event bus topics for `qa_turn.created`, `interview.finished`,
  `task.enqueued`, `interview.scoring_started`, `interview.reported`, and `task.completed`.
- Behavior signal module with explicit administrator enablement and candidate consent gates.
- BehaviorSignal schema forbids extra personality, emotion, reliability, or similar derived fields.
- Candidate behavior-signal consent can be revoked and blocks future signal-enabled interviews.
- Realtime signal emission re-checks active consent, so revoked consent suppresses later WS hints.
- Configurable LLM client with mock mode and OpenAI-compatible HTTP mode.
- LLM JSON responses are pydantic-validated with configurable retry before deterministic fallback.
- Async and sync LLM paths support injectable transports for network-free unit tests.
- Optional LLM provider-call rate limiting is enforced before outbound model requests.
- Safe runtime config and LLM smoke-test scripts.
- Prometheus-style `/metrics` endpoint for local HTTP request counters and duration sums.
- `/metrics` also exposes domain/task event counters for the offline scoring path.
- Structured JSON request logs with propagated `X-Request-ID` correlation IDs.
- W3C `traceparent` propagation with trace/span IDs included in request logs.
- Optional per-client gateway rate limit gate with local and Redis-backed counter modes.
- Docker Compose entrypoint for gateway, PostgreSQL-backed persistence, Redis, and MinIO.

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
