from __future__ import annotations

from libs.schemas import AIGCResult, DimensionScore, EvidenceRef, InterviewContext, InterviewScore


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


def score_interview(ctx: InterviewContext, aigc_results: list[AIGCResult]) -> InterviewScore:
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

