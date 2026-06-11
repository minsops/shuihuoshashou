from __future__ import annotations

from libs.common.prompts import load_prompt
from libs.llm_client import LLMMessage, get_llm_client
from libs.schemas import CredibilitySignal, ProbeChain, ProbeRequest, ProbeResponse, ProbeSuggestion
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
        chain = _chain_for_suggestion(request.probe_chains, index=index, competency=item.name)
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
                chain_id=chain.chain_id if chain is not None else None,
                chain_label=_chain_label(chain) if chain is not None else None,
            )
        )
    return ProbeResponse(suggestions=suggestions, credibility=credibility)


async def generate_probe(request: ProbeRequest) -> ProbeResponse:
    fallback = fallback_probe(request)
    messages = [
        LLMMessage(role="system", content=load_prompt("probe_system.md")),
        LLMMessage(role="user", content=request.model_dump_json()),
    ]
    draft = await get_llm_client().complete_json(messages, ProbeResponse, fallback)
    return _normalize_probe_response(draft, fallback)


def _normalize_probe_response(response: ProbeResponse, fallback: ProbeResponse | None = None) -> ProbeResponse:
    suggestions = [
        _merge_fallback_chain_fields(suggestion, fallback, index).model_copy(update={"priority": index})
        for index, suggestion in enumerate(
            sorted(response.suggestions, key=lambda item: item.priority),
            start=1,
        )
    ]
    return ProbeResponse(suggestions=suggestions, credibility=response.credibility)


def _chain_for_suggestion(
    chains: list[ProbeChain],
    *,
    index: int,
    competency: str,
) -> ProbeChain | None:
    if not chains:
        return None
    unresolved = [chain for chain in chains if chain.verdict == "unresolved"]
    candidates = unresolved or chains
    direct_match = next(
        (chain for chain in candidates if competency.lower() in chain.topic.lower()),
        None,
    )
    if direct_match is not None:
        return direct_match
    if competency == "项目真实性":
        resume_chain = next((chain for chain in candidates if chain.origin == "resume_claim"), None)
        if resume_chain is not None:
            return resume_chain
    if index - 1 < len(candidates):
        return candidates[index - 1]
    return candidates[0]


def _chain_label(chain: ProbeChain) -> str:
    origin_label = {
        "resume_claim": "简历核验链",
        "answer_claim": "回答声明链",
        "competency_gap": "能力缺口链",
    }[chain.origin]
    return f"{origin_label} · 第 {len(chain.links) + 1} 层"


def _merge_fallback_chain_fields(
    suggestion: ProbeSuggestion,
    fallback: ProbeResponse | None,
    priority: int,
) -> ProbeSuggestion:
    if suggestion.chain_id and suggestion.chain_label:
        return suggestion
    if fallback is None:
        return suggestion
    fallback_suggestion = next(
        (item for item in fallback.suggestions if item.priority == priority),
        None,
    )
    if fallback_suggestion is None:
        return suggestion
    return suggestion.model_copy(
        update={
            "chain_id": suggestion.chain_id or fallback_suggestion.chain_id,
            "chain_label": suggestion.chain_label or fallback_suggestion.chain_label,
        }
    )
