from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from libs.common.config import get_settings
from libs.common.prompts import load_prompt
from libs.common.textsim import cosine_similarity
from libs.llm_client import LLMMessage, get_llm_client
from libs.schemas import AIGCResult, ProbeChain, QATurn


TEMPLATE_PATH = Path(__file__).with_name("templates") / "common_answer_templates.txt"


class AIGCReviewItem(BaseModel):
    turn_id: str
    ai_generated_prob: float = Field(ge=0.0, le=1.0)
    reason: str


class AIGCReview(BaseModel):
    results: list[AIGCReviewItem] = Field(default_factory=list)


def llm_review_aigc(turns: list[QATurn], results: list[AIGCResult]) -> list[AIGCResult]:
    """用大模型逐条评判回答是否疑似 AI 生成/背稿，与确定性检测取较高风险合并。"""
    if not turns or not results:
        return results
    messages = [
        LLMMessage(role="system", content=load_prompt("aigc_review.md")),
        LLMMessage(
            role="user",
            content=json.dumps(
                {
                    "turns": [
                        {
                            "turn_id": turn.turn_id,
                            "question": turn.question,
                            "answer": turn.answer,
                        }
                        for turn in turns
                    ]
                },
                ensure_ascii=False,
            ),
        ),
    ]
    review = get_llm_client().complete_json_sync(messages, AIGCReview, AIGCReview())
    if not review.results:
        return results
    by_turn = {item.turn_id: item for item in review.results}
    settings = get_settings()
    merged: list[AIGCResult] = []
    for result in results:
        item = by_turn.get(result.turn_id)
        if item is None:
            merged.append(result)
            continue
        prob = max(result.ai_generated_prob, round(item.ai_generated_prob, 3))
        merged.append(
            result.model_copy(
                update={
                    "ai_generated_prob": prob,
                    "flagged": result.flagged or prob >= settings.aigc_ai_prob_threshold,
                    "llm_reason": item.reason.strip() or None,
                }
            )
        )
    return merged


@lru_cache
def load_templates() -> tuple[str, ...]:
    return tuple(
        line.strip()
        for line in TEMPLATE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    )


def detect_turn(
    turn: QATurn,
    transport: httpx.BaseTransport | None = None,
    *,
    cracked_turn_ids: set[str] | None = None,
) -> AIGCResult:
    local = _local_detect_turn(turn, cracked_turn_ids=cracked_turn_ids or set())
    settings = get_settings()
    if settings.aigc_detector_provider != "http":
        return local
    try:
        return _http_detect_turn(turn, local, transport)
    except (httpx.HTTPError, KeyError, TypeError, ValueError, RuntimeError):
        return local


def _local_detect_turn(turn: QATurn, *, cracked_turn_ids: set[str]) -> AIGCResult:
    answer = turn.answer.strip()
    templates = load_templates()
    max_template = max(templates, key=lambda template: cosine_similarity(answer, template))
    template_similarity = cosine_similarity(answer, max_template)
    polished_markers = ["首先", "其次", "最后", "综上", "显著提升", "业务痛点"]
    ai_generated_prob = min(
        1.0,
        0.15
        + 0.15 * sum(marker in answer for marker in polished_markers)
        + (0.25 if len(answer) > 180 and "我" not in answer[:80] else 0.0)
        + template_similarity * 0.4,
    )
    settings = get_settings()
    # Voice rehearsal detection uses transcript text features only. It does not infer biometric
    # traits or facial/behavioral signals.
    fluency_anomaly = _fluency_anomaly(answer, polished_markers)
    chain_crack_bonus = 1.0 if turn.turn_id in cracked_turn_ids else 0.0
    rehearsal_score = min(
        1.0,
        0.5 * template_similarity + 0.3 * fluency_anomaly + 0.2 * chain_crack_bonus,
    )
    flagged = (
        rehearsal_score >= settings.rehearsal_threshold
        or template_similarity >= settings.aigc_template_similarity_threshold
    )
    return AIGCResult(
        turn_id=turn.turn_id,
        ai_generated_prob=round(ai_generated_prob, 3),
        template_similarity=round(template_similarity, 3),
        rehearsal_score=round(rehearsal_score, 3),
        mode="voice",
        matched_template=max_template if template_similarity > 0.2 else None,
        flagged=flagged,
    )


def _fluency_anomaly(answer: str, polished_markers: list[str]) -> float:
    if not answer:
        return 0.0
    filler_count = sum(answer.count(marker) for marker in ("嗯", "呃", "这个", "然后"))
    marker_score = min(1.0, sum(marker in answer for marker in polished_markers) / 3)
    smooth_delivery = 1.0 if len(answer) >= 24 and filler_count == 0 else 0.0
    punctuation_density = min(1.0, sum(answer.count(item) for item in ("，", "。", "；")) / 8)
    return round(
        min(1.0, 0.45 * marker_score + 0.4 * smooth_delivery + 0.15 * punctuation_density),
        3,
    )


def _http_detect_turn(
    turn: QATurn,
    local: AIGCResult,
    transport: httpx.BaseTransport | None,
) -> AIGCResult:
    settings = get_settings()
    if not settings.aigc_detector_base_url:
        raise RuntimeError("AIGC_DETECTOR_PROVIDER=http requires AIGC_DETECTOR_BASE_URL")
    url = (
        settings.aigc_detector_base_url.rstrip("/")
        + "/"
        + settings.aigc_detector_api_path.lstrip("/")
    )
    payload = {
        "turn_id": turn.turn_id,
        "question": turn.question,
        "answer": turn.answer,
        "local_template_similarity": local.template_similarity,
        "local_matched_template": local.matched_template,
    }
    with httpx.Client(timeout=settings.aigc_detector_timeout_seconds, transport=transport) as client:
        response = client.post(url, headers=_aigc_headers(), json=payload)
        response.raise_for_status()
    data = response.json()
    probability = max(0.0, min(1.0, float(_extract_path(data, settings.aigc_detector_probability_path))))
    flagged_value = _extract_optional(data, settings.aigc_detector_flagged_path, None)
    flagged = (
        _coerce_bool(flagged_value)
        if flagged_value is not None
        else probability >= settings.aigc_ai_prob_threshold
    )
    flagged = (
        flagged
        or probability >= settings.aigc_ai_prob_threshold
        or local.rehearsal_score >= settings.rehearsal_threshold
        or local.template_similarity >= settings.aigc_template_similarity_threshold
    )
    return AIGCResult(
        turn_id=turn.turn_id,
        ai_generated_prob=round(probability, 3),
        template_similarity=local.template_similarity,
        rehearsal_score=local.rehearsal_score,
        mode=local.mode,
        matched_template=local.matched_template,
        flagged=flagged,
    )


def _aigc_headers() -> dict[str, str]:
    settings = get_settings()
    if not settings.aigc_detector_api_key:
        return {}
    auth_value = (
        f"{settings.aigc_detector_auth_scheme} {settings.aigc_detector_api_key}"
        if settings.aigc_detector_auth_scheme
        else settings.aigc_detector_api_key
    )
    return {settings.aigc_detector_auth_header: auth_value}


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


def _extract_optional(payload: Any, path: str, fallback: Any) -> Any:
    try:
        return _extract_path(payload, path)
    except (KeyError, IndexError, TypeError, ValueError):
        return fallback


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def detect_interview(
    turns: list[QATurn],
    transport: httpx.BaseTransport | None = None,
    *,
    probe_chains: list[ProbeChain] | None = None,
) -> list[AIGCResult]:
    cracked_turn_ids = {
        link.answer_turn_id
        for chain in probe_chains or []
        if chain.verdict == "cracked"
        for link in chain.links
    }
    return [
        detect_turn(turn, transport=transport, cracked_turn_ids=cracked_turn_ids)
        for turn in turns
    ]
