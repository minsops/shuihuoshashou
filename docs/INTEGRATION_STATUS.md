# Integration Status

## Current State

The local implementation is complete as a runnable MVP:

- FastAPI gateway and API docs.
- Local demo UI at `/`.
- One-shot offline evaluation at `/api/offline/evaluate`.
- WebSocket real-time text/audio-stub probe flow.
- Pydantic v2 shared schemas.
- SQLite local persistence.
- Probe generation, scoring, AIGC/template checks, consistency checks, HTML/PDF report generation.
- Behavior signal module with explicit candidate consent gate.
- Configurable LLM client with mock mode and OpenAI-compatible HTTP mode.
- Safe runtime config and LLM smoke-test scripts.

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

## Current Blocker

Real MiMo API smoke testing now reaches the current endpoint, but the provided API key is rejected by
the service. Multiple common OpenAI-compatible auth header variants and model-name variants were
tested; all returned the same `HTTP 401 Invalid API Key`.

Observed command:

```bash
python scripts/diagnose_llm_network.py
```

Observed result:

```text
base_url: https://token-plan-cn.xiaomimimo.com/v1
host: token-plan-cn.xiaomimimo.com
port: 443
dns: ok
tcp: ok
tls: ok
LLM smoke test failed: HTTP 401 Invalid API Key
```

The endpoint, DNS, TCP, TLS, request path, and response-path plumbing are in place. A fresh valid API
key is needed for the final live smoke test.

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

## Next Step To Unblock

Use a fresh valid MiMo API key. Then run:

```bash
python scripts/diagnose_llm_network.py
LLM_PROVIDER=openai_compatible LLM_API_KEY=<new-key> python scripts/check_llm.py
LLM_PROVIDER=openai_compatible LLM_API_KEY=<new-key> python scripts/diagnose_llm_auth.py
```
