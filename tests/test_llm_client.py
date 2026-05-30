from __future__ import annotations

import json

import httpx

from libs.common.config import get_settings
from libs.llm_client import LLMClient, LLMMessage
from libs.schemas import CredibilitySignal, ProbeResponse, ProbeSuggestion


def _fallback() -> ProbeResponse:
    return ProbeResponse(
        suggestions=[
            ProbeSuggestion(
                question="fallback",
                target="fallback",
                competency="fallback",
                priority=1,
            )
        ],
        credibility=CredibilitySignal(level="vague", reason="fallback", drill_down_hint="fallback"),
    )


async def _call_with_transport(transport: httpx.MockTransport) -> ProbeResponse:
    client = LLMClient(transport=transport)
    return await client.complete_json(
        [LLMMessage(role="user", content="hello")],
        ProbeResponse,
        _fallback(),
    )


def test_llm_client_uses_configurable_protocol(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example.test/api")
    monkeypatch.setenv("LLM_API_PATH", "/v2/chat")
    monkeypatch.setenv("LLM_API_KEY", "secret")
    monkeypatch.setenv("LLM_AUTH_HEADER", "X-API-Key")
    monkeypatch.setenv("LLM_AUTH_SCHEME", "")
    monkeypatch.setenv("LLM_RESPONSE_CONTENT_PATH", "data.output")
    monkeypatch.setenv("LLM_EXTRA_BODY_JSON", '{"custom":{"enabled":true}}')
    get_settings.cache_clear()

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://llm.example.test/api/v2/chat"
        assert request.headers["X-API-Key"] == "secret"
        body = json.loads(request.content)
        assert body["model"] == "mimo-v2.5-pro"
        assert body["custom"]["enabled"] is True
        return httpx.Response(
            200,
            json={
                "data": {
                    "output": json.dumps(
                        {
                            "suggestions": [
                                {
                                    "question": "真实接口追问",
                                    "target": "验证项目真实性",
                                    "competency": "项目真实性",
                                    "priority": 1,
                                }
                            ],
                            "credibility": {
                                "level": "solid",
                                "reason": "包含细节",
                                "drill_down_hint": "继续追问异常处理",
                            },
                        },
                        ensure_ascii=False,
                    )
                }
            },
        )

    response = __import__("asyncio").run(_call_with_transport(httpx.MockTransport(handler)))
    assert response.suggestions[0].question == "真实接口追问"
    assert response.credibility.level == "solid"


def test_llm_client_falls_back_on_bad_response(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example.test")
    monkeypatch.setenv("LLM_API_KEY", "secret")
    get_settings.cache_clear()

    transport = httpx.MockTransport(lambda _: httpx.Response(500, json={"error": "boom"}))
    response = __import__("asyncio").run(_call_with_transport(transport))
    assert response.suggestions[0].question == "fallback"


def test_llm_client_retries_once_after_bad_json(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example.test")
    monkeypatch.setenv("LLM_API_KEY", "secret")
    monkeypatch.setenv("LLM_MAX_RETRIES", "1")
    get_settings.cache_clear()
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(200, json={"choices": [{"message": {"content": "not-json"}}]})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "suggestions": [
                                        {
                                            "question": "重试后成功",
                                            "target": "验证项目真实性",
                                            "competency": "项目真实性",
                                            "priority": 1,
                                        }
                                    ],
                                    "credibility": {
                                        "level": "solid",
                                        "reason": "重试解析成功",
                                        "drill_down_hint": "继续追问",
                                    },
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            },
        )

    response = __import__("asyncio").run(_call_with_transport(httpx.MockTransport(handler)))

    assert calls == 2
    assert response.suggestions[0].question == "重试后成功"


def test_llm_client_retry_count_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example.test")
    monkeypatch.setenv("LLM_API_KEY", "secret")
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    get_settings.cache_clear()
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": "not-json"}}]})

    response = __import__("asyncio").run(_call_with_transport(httpx.MockTransport(handler)))

    assert calls == 1
    assert response.suggestions[0].question == "fallback"


def test_llm_client_can_raise_safe_error(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example.test")
    monkeypatch.setenv("LLM_API_KEY", "secret")
    get_settings.cache_clear()

    transport = httpx.MockTransport(lambda _: httpx.Response(401, json={"error": "bad key"}))
    client = LLMClient(transport=transport)
    try:
        __import__("asyncio").run(
            client.complete_json(
                [LLMMessage(role="user", content="hello")],
                ProbeResponse,
                _fallback(),
                raise_on_error=True,
            )
        )
    except RuntimeError as exc:
        assert "HTTP 401" in str(exc)
        assert "secret" not in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
