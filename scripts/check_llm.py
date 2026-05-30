from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


async def main() -> int:
    from libs.common.runtime import get_runtime_status
    from libs.llm_client import LLMMessage, get_llm_client
    from libs.schemas import CredibilitySignal, ProbeResponse, ProbeSuggestion

    status = get_runtime_status()
    print(status.model_dump_json(indent=2))
    fallback = ProbeResponse(
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
    try:
        response = await get_llm_client().complete_json(
            [
                LLMMessage(
                    role="system",
                    content=(
                        "Return only a JSON object with this exact shape: "
                        '{"suggestions":[{"question":"...","target":"...","competency":"...",'
                        '"priority":1}],"credibility":{"level":"solid","reason":"...",'
                        '"drill_down_hint":"..."}}. The credibility.level must be one of '
                        "solid, vague, suspicious."
                    ),
                ),
                LLMMessage(
                    role="user",
                    content="Generate one interview probe question for a vague backend project answer.",
                ),
            ],
            ProbeResponse,
            fallback,
            raise_on_error=status.llm_provider != "mock",
        )
    except RuntimeError as exc:
        print(f"LLM smoke test failed: {exc}")
        return 1
    if response == fallback and status.llm_provider != "mock":
        print("LLM smoke test fell back. Check provider protocol, base URL, auth, and response path.")
        return 1
    print("LLM smoke test ok.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
