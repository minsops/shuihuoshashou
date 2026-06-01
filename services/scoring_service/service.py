from __future__ import annotations

import json
from typing import Literal

from libs.common.prompts import load_prompt
from libs.llm_client import LLMMessage, get_llm_client
from libs.schemas import AIGCResult, DimensionScore, EvidenceRef, InterviewContext, InterviewScore, QATurn


def _evidence_for_dimension(ctx: InterviewContext, dimension: str) -> list[EvidenceRef]:
    refs: list[EvidenceRef] = []
    for turn in ctx.turns[:3]:
        refs.append(
            EvidenceRef(
                turn_id=turn.turn_id,
                quote_start_ms=turn.answer_start_ms,
                quote_end_ms=turn.answer_end_ms,
                excerpt=turn.answer[:120],
            )
        )
    return refs


def fallback_score_interview(
    ctx: InterviewContext,
    aigc_results: list[AIGCResult],
) -> InterviewScore:
    _ensure_scoreable_context(ctx)
    risk_penalty = 0.0
    risk_notes: list[str] = []
    if ctx.flags:
        risk_penalty += 12.0
        risk_notes.extend(flag.description for flag in ctx.flags)
    flagged_aigc = [item for item in aigc_results if item.flagged]
    if flagged_aigc:
        risk_penalty += min(18.0, 6.0 * len(flagged_aigc))
        risk_notes.append("部分回答疑似模板化或 AI 生成，需要人工复核。")

    dimensions: list[DimensionScore] = []
    for item in ctx.competency_model.items:
        base = 78.0
        if item.name == "项目真实性":
            base -= risk_penalty * 0.7
        elif item.name == "注水风险":
            base = max(0.0, 100.0 - risk_penalty * 3)
        elif item.name == "沟通与逻辑":
            avg_len = sum(len(turn.answer) for turn in ctx.turns) / max(1, len(ctx.turns))
            base += 5.0 if avg_len > 80 else -8.0
        dimensions.append(
            DimensionScore(
                dimension=item.name,
                score=round(max(0.0, min(100.0, base)), 2),
                weight=item.weight,
                evidence=_evidence_for_dimension(ctx, item.name),
            )
        )

    positive = [d for d in dimensions if d.weight > 0]
    weight_sum = sum(d.weight for d in positive) or 1.0
    total = sum(d.score * d.weight for d in positive) / weight_sum
    total -= risk_penalty * 0.35
    total = round(max(0.0, min(100.0, total)), 2)
    if total >= 88:
        recommendation = "strong_yes"
    elif total >= 75:
        recommendation = "yes"
    elif total >= 60:
        recommendation = "hold"
    else:
        recommendation = "no"
    return InterviewScore(
        session_id=ctx.session_id,
        dimensions=dimensions,
        total_score=total,
        risk_notes=risk_notes,
        recommendation=recommendation,
    )


def score_interview(ctx: InterviewContext, aigc_results: list[AIGCResult]) -> InterviewScore:
    _ensure_scoreable_context(ctx)
    fallback = fallback_score_interview(ctx, aigc_results)
    messages = [
        LLMMessage(role="system", content=load_prompt("scoring_system.md")),
        LLMMessage(
            role="user",
            content=_scoring_payload(ctx, aigc_results),
        ),
    ]
    draft = get_llm_client().complete_json_sync(messages, InterviewScore, fallback)
    return _normalize_score(ctx, draft, fallback)


def _scoring_payload(ctx: InterviewContext, aigc_results: list[AIGCResult]) -> str:
    payload = {
        "job_id": ctx.job_id,
        "candidate_id": ctx.candidate_id,
        "competency_model": ctx.competency_model.model_dump(),
        "turns": [turn.model_dump() for turn in ctx.turns],
        "consistency_flags": [flag.model_dump() for flag in ctx.flags],
        "aigc_results": [item.model_dump() for item in aigc_results],
        "instructions": (
            "Return JSON matching InterviewScore. Provide one DimensionScore per competency "
            "dimension with evidence referencing existing turn_id values. Python will recompute "
            "total_score from dimension scores and weights."
        ),
    }
    return json.dumps(payload, ensure_ascii=False, default=str)


def _ensure_scoreable_context(ctx: InterviewContext) -> None:
    if not ctx.turns:
        raise ValueError("cannot score interview without candidate turns")


def _normalize_score(
    ctx: InterviewContext,
    draft: InterviewScore,
    fallback: InterviewScore,
) -> InterviewScore:
    turns_by_id = {turn.turn_id: turn for turn in ctx.turns}
    fallback_by_dimension = {dimension.dimension: dimension for dimension in fallback.dimensions}
    normalized_dimensions: list[DimensionScore] = []
    for item in ctx.competency_model.items:
        draft_dimension = next(
            (dimension for dimension in draft.dimensions if dimension.dimension == item.name),
            None,
        )
        if draft_dimension is None:
            normalized_dimensions.append(fallback_by_dimension[item.name])
            continue
        evidence = _normalize_evidence_refs(draft_dimension.evidence, turns_by_id)
        if not evidence:
            evidence = fallback_by_dimension[item.name].evidence
        normalized_dimensions.append(
            DimensionScore(
                dimension=item.name,
                score=round(max(0.0, min(100.0, draft_dimension.score)), 2),
                weight=item.weight,
                evidence=evidence,
            )
        )

    total = _compute_total_score(normalized_dimensions)
    return InterviewScore(
        session_id=ctx.session_id,
        dimensions=normalized_dimensions,
        total_score=total,
        risk_notes=draft.risk_notes or fallback.risk_notes,
        recommendation=_recommendation(total),
    )


def _normalize_evidence_refs(
    refs: list[EvidenceRef],
    turns_by_id: dict[str, QATurn],
) -> list[EvidenceRef]:
    normalized: list[EvidenceRef] = []
    for ref in refs:
        turn = turns_by_id.get(ref.turn_id)
        if turn is None or not ref.excerpt.strip():
            continue
        answer = turn.answer
        excerpt = ref.excerpt.strip()
        if excerpt not in answer:
            excerpt = answer[:120]
        answer_start_ms = turn.answer_start_ms
        answer_end_ms = turn.answer_end_ms
        quote_start_ms = max(answer_start_ms, min(ref.quote_start_ms, answer_end_ms))
        quote_end_ms = max(quote_start_ms, min(ref.quote_end_ms, answer_end_ms))
        normalized.append(
            EvidenceRef(
                turn_id=ref.turn_id,
                quote_start_ms=quote_start_ms,
                quote_end_ms=quote_end_ms,
                excerpt=excerpt,
            )
        )
    return normalized


def _compute_total_score(dimensions: list[DimensionScore]) -> float:
    positive = [dimension for dimension in dimensions if dimension.weight > 0]
    weight_sum = sum(dimension.weight for dimension in positive) or 1.0
    total = sum(dimension.score * dimension.weight for dimension in positive) / weight_sum
    for dimension in dimensions:
        if dimension.weight < 0:
            total -= (100.0 - dimension.score) * abs(dimension.weight)
    return round(max(0.0, min(100.0, total)), 2)


def _recommendation(total_score: float) -> Literal["strong_yes", "yes", "hold", "no"]:
    if total_score >= 88:
        return "strong_yes"
    if total_score >= 75:
        return "yes"
    if total_score >= 60:
        return "hold"
    return "no"
