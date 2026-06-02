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
- WebSocket text turns with blank answers are rejected before local ASR fallback can create placeholder
  transcripts.
- Optional WebSocket audio metadata is validated against the startup contract: PCM/Opus-style
  payloads, 16 kHz sample rate, and one channel.
- Configurable real-time probe trigger rules for candidate answer length, drill-down topic matching,
  minimum interval, and interviewer-initiated manual probes.
- Configurable HTTP cloud ASR adapter behind the `ASREngine` interface.
- HTTP ASR responses normalize string finality flags, treat provisional labels as non-final, clamp
  confidence values, and enforce non-negative monotonic timestamp ranges.
- Shared TranscriptSegment, QATurn, and EvidenceRef schemas reject negative or inverted timestamp ranges.
- Shared TranscriptSegment schemas also reject blank `session_id` and transcript text so empty ASR
  output cannot enter orchestration.
- ASR session manager supports partial-to-final chunk updates, repeated-final deduplication,
  stale-final rejection, local audio-cluster speaker resolution, and conservative short-gap
  speaker continuity smoothing.
- Speaker diarization is pluggable: local deterministic audio clusters for development, or an
  HTTP provider for production voice embedding / cloud diarization services.
- Docker Compose forwards `SPEAKER_DIARIZATION_*` settings to the gateway, and gateway startup
  reloads the ASR session diarizer from runtime configuration.
- Separate `credibility` WebSocket event after probe generation.
- Pydantic v2 shared schemas.
- SQLite local persistence.
- PostgreSQL core schema SQL for compose initialization.
- PostgreSQL schema-level CHECK constraints enforce core state, timestamp, score, AIGC, report, and
  consent invariants at the storage boundary.
- Database URL dialect detection and runtime connection support for SQLite and PostgreSQL targets.
- SQLite startup migrations backfill later realtime/report columns so older local demo databases stay usable.
- Local SQLite service code validates interview/consent candidate references to match PostgreSQL
  foreign-key behavior.
- Separate `qa_turns` persistence for auditable answer evidence.
- Shared QATurn and EvidenceRef schemas reject blank question, answer, and excerpt text.
- Shared job, candidate, competency, and probe-card schemas reject blank required text.
- JD knowledge base indexes competency-specific probe patterns with deterministic embeddings and optional pgvector nearest-neighbor search.
- JD competency generation uses the shared LLM JSON client with `prompts/competency_gen.md` and
  deterministic fallback, then normalizes outputs to preserve required scoring dimensions and
  weights.
- ProbeResponse schema validation enforces the one-to-three suggestion contract for realtime
  interviewer cards.
- Probe generation, scoring, AIGC/template checks, consistency checks, HTML/PDF report generation.
- Internal HTTP-style contracts for standalone probe, AIGC detection, scoring, and report calls.
- Scoring uses the shared LLM JSON client for structured dimension drafts, then normalizes evidence
  and recomputes final totals in Python.
- Offline scoring requires at least one candidate turn, and each DimensionScore requires at least
  one EvidenceRef.
- Scoring evidence is re-anchored to persisted turns so fabricated excerpts or out-of-range
  timestamps from LLM drafts do not enter reports.
- Runtime probe prompts are kept under `prompts/` and loaded by services instead of being embedded in code.
- AIGC checks use a local corpus with character n-gram cosine similarity plus optional HTTP detector
  integration; the final flag applies configured probability/template thresholds with deterministic
  local fallback.
- Interview context persists extracted fact claims for role, responsibility, technology, and metric
  statements, and consistency checks run against that fact table.
- Structured JSON and HTML/PDF reports include highlights, radar charts, AIGC risk highlights,
  recommendation, and the full interview transcript.
- PDF generation falls back to an auditable text PDF with score, recommendation, dimension evidence,
  risk, AIGC, and transcript summary if WeasyPrint's native rendering stack is unavailable.
- Report input validation rejects score/session mismatches, dimension or weight mismatches,
  inconsistent totals/recommendations, unknown EvidenceRef/AIGC turn ids, out-of-range evidence
  timestamps, evidence excerpts not present in the transcript answer, and incomplete or duplicate
  AIGC coverage; unflagged AIGC results above configured thresholds are also rejected.
- Report artifacts support local `file://` storage and SigV4 uploads to S3-compatible storage for
  structured report JSON, HTML, PDF, and transcript JSON outputs.
- Gateway exposes report JSON, HTML, PDF, and transcript JSON artifact endpoints.
- Local offline scoring task flow with `FINISHED -> SCORING -> REPORTED` state transitions.
- State guards prevent scoring before `FINISHED` and prevent turn edits or restarts after reporting.
- Local task queue boundary for offline scoring with Redis Streams task publication and worker consumption.
- Optional Celery task publication and worker registration for `interview.offline_scoring`.
- Celery task publisher and offline worker share configurable `CELERY_TASK_QUEUE` routing so async
  report generation lands on the queue the worker consumes.
- Docker Compose includes an optional `worker` profile for running the Celery offline-scoring worker.
- Configurable async end-interview mode returns queued task metadata while workers generate reports.
- Async end-interview advances queued interviews to `SCORING` and rejects duplicate queueing.
- In-memory event bus topics for `qa_turn.created`, `interview.finished`,
  `task.enqueued`, `task.completed`, `task.failed`, `task.worker_completed`,
  `task.worker_failed`, `interview.scoring_started`, and `interview.reported`.
- Behavior signal module with explicit administrator enablement and candidate consent gates.
- BehaviorSignal schema forbids extra personality, emotion, reliability, or similar derived fields.
- Candidate behavior-signal consent can be revoked and blocks future signal-enabled interviews, with
  SQLite/PostgreSQL-compatible boolean predicates.
- Realtime signal emission re-checks active consent, so revoked consent suppresses later WS hints.
- Configurable LLM client with mock mode and OpenAI-compatible HTTP mode.
- LLM JSON responses are pydantic-validated from string or object payloads with configurable retry
  before deterministic fallback.
- Async and sync LLM paths support injectable transports for network-free unit tests.
- Optional LLM provider-call rate limiting is enforced before outbound model requests.
- Safe runtime config and LLM smoke-test scripts.
- Runtime config status exposes non-secret provider paths, response mapping paths, and timeouts for deployment diagnostics.
- Runtime config status redacts database URL passwords.
- Prometheus-style `/metrics` endpoint for local HTTP request counters and duration sums.
- `/metrics` also exposes domain/task event counters, including offline scoring success and
  failure events.
- Structured JSON request logs rendered through structlog with propagated `X-Request-ID`
  correlation IDs.
- W3C `traceparent` propagation with trace/span IDs included in request logs.
- Optional OpenTelemetry OTLP HTTP tracing is wired into the FastAPI gateway when
  `OTEL_EXPORTER_OTLP_ENDPOINT` is configured and the `otel` extra is installed.
- Optional per-client gateway rate limit gate with local and Redis-backed counter modes.
- Docker Compose entrypoint for gateway, PostgreSQL-backed persistence, Redis, and MinIO.
- Docker Compose forwards gateway auth, LLM retry/rate-limit, ASR mapping, probe trigger, AIGC
  detector, Redis rate-limit, and worker AIGC/LLM runtime settings into the services that use them.
- Docker Compose mounts only the core PostgreSQL schema into initdb by default, leaving the optional
  pgvector migration to `JD_VECTOR_BACKEND=pgvector` deployments with the extension installed.

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
