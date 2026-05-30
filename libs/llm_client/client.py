from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from libs.common.config import get_settings

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class LLMMessage:
    role: str
    content: str


class LLMClient:
    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._transport = transport

    async def complete_json(
        self,
        messages: list[LLMMessage],
        schema: type[T],
        fallback: T,
        raise_on_error: bool = False,
    ) -> T:
        settings = get_settings()
        if settings.llm_provider == "mock" or not settings.llm_api_key or not settings.llm_base_url:
            return fallback
        payload = {
            "model": settings.llm_model,
            "messages": [message.__dict__ for message in messages],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }
        if settings.llm_extra_body_json:
            extra_body = json.loads(settings.llm_extra_body_json)
            if not isinstance(extra_body, dict):
                raise ValueError("LLM_EXTRA_BODY_JSON must decode to an object")
            payload.update(extra_body)
        auth_value = (
            f"{settings.llm_auth_scheme} {settings.llm_api_key}"
            if settings.llm_auth_scheme
            else settings.llm_api_key
        )
        headers = {settings.llm_auth_header: auth_value}
        url = settings.llm_base_url.rstrip("/") + "/" + settings.llm_api_path.lstrip("/")
        last_error: Exception | None = None
        max_attempts = max(1, settings.llm_max_retries + 1)
        for attempt in range(max_attempts):
            try:
                async with httpx.AsyncClient(
                    timeout=settings.llm_timeout_seconds,
                    transport=self._transport,
                ) as client:
                    response = await client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                content = _extract_path(response.json(), settings.llm_response_content_path)
                if not isinstance(content, str):
                    raise ValueError("LLM response content path did not resolve to a string")
                data: Any = json.loads(content)
                return schema.model_validate(data)
            except Exception as exc:
                last_error = exc
                if attempt == max_attempts - 1:
                    if raise_on_error:
                        raise RuntimeError(_safe_error_message(last_error)) from exc
                    return fallback
        return fallback


def get_llm_client() -> LLMClient:
    return LLMClient()


def _extract_path(payload: Any, path: str) -> Any:
    value = payload
    for part in path.split("."):
        if isinstance(value, list):
            value = value[int(part)]
        elif isinstance(value, dict):
            value = value[part]
        else:
            raise KeyError(path)
    return value


def _safe_error_message(error: Exception | None) -> str:
    if error is None:
        return "unknown LLM error"
    if isinstance(error, httpx.HTTPStatusError):
        response = error.response
        body = response.text[:500]
        return f"HTTP {response.status_code}: {body}"
    detail = str(error) or repr(error)
    return f"{type(error).__name__}: {detail[:500]}"
