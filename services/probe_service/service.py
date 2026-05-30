from __future__ import annotations

from libs.llm_client import LLMMessage, get_llm_client
from libs.schemas import CredibilitySignal, ProbeRequest, ProbeResponse, ProbeSuggestion
from services.jd_kb_service.service import retrieve_job_probe_patterns, retrieve_probe_patterns


VAGUE_MARKERS = ["负责", "参与", "优化", "提升", "很多", "比较", "主要", "一些"]
DETAIL_MARKERS = ["因为", "具体", "我写", "指标", "%", "ms", "qps", "故障", "排查", "取舍"]


def assess_credibility(answer: str) -> CredibilitySignal:
    vague = sum(marker in answer for marker in VAGUE_MARKERS)
    detail = sum(marker in answer.lower() for marker in DETAIL_MARKERS)
    if len(answer.strip()) < 30 or vague >= 3 and detail <= 1:
        return CredibilitySignal(
            level="suspicious",
            reason="回答偏概括，缺少可验证的个人贡献、数据或异常处理细节。",
            drill_down_hint="要求候选人说明自己写了哪部分、关键决策依据和一次真实故障。",
        )
    if vague >= 2 and detail <= 2:
        return CredibilitySignal(
            level="vague",
            reason="回答有一定信息，但关键实现和结果证据仍不充分。",
            drill_down_hint="沿项目细节、指标口径和技术取舍继续下钻。",
        )
    return CredibilitySignal(
        level="solid",
        reason="回答包含较多细节或可验证线索。",
        drill_down_hint="追问边界条件、失败案例和复盘，验证稳定性。",
    )


def fallback_probe(request: ProbeRequest) -> ProbeResponse:
    credibility = assess_credibility(request.latest_answer)
    suggestions: list[ProbeSuggestion] = []
    query = f"{request.latest_answer} {credibility.drill_down_hint}"
    pattern_hits = retrieve_job_probe_patterns(request.job_id, query, limit=3)
    if not pattern_hits:
        pattern_hits = retrieve_probe_patterns(request.competency_model, query, limit=3)
    for index, item in enumerate(request.competency_model.items[:3], start=1):
        hit = next((candidate for candidate in pattern_hits if candidate.competency == item.name), None)
        question = (
            hit.pattern
            if hit is not None
            else (
                f"围绕「{item.name}」，请候选人讲一个最近项目中的具体细节："
                "他本人负责哪一段、为什么这样设计、遇到过什么异常以及最后指标如何变化？"
            )
        )
        suggestions.append(
            ProbeSuggestion(
                question=question,
                target="验证项目真实性" if item.name == "项目真实性" else "测试能力深度",
                competency=item.name,
                priority=index,
            )
        )
    return ProbeResponse(suggestions=suggestions, credibility=credibility)


async def generate_probe(request: ProbeRequest) -> ProbeResponse:
    fallback = fallback_probe(request)
    messages = [
        LLMMessage(role="system", content="Return strict JSON matching ProbeResponse."),
        LLMMessage(role="user", content=request.model_dump_json()),
    ]
    return await get_llm_client().complete_json(messages, ProbeResponse, fallback)
