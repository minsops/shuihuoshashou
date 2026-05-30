from __future__ import annotations

import re

from libs.schemas import AIGCResult, QATurn


TEMPLATES = [
    "我主要负责整体架构设计并推动项目落地最终取得显著提升",
    "通过深入分析业务痛点优化流程提升效率降低成本",
    "我具备良好的沟通能力学习能力和抗压能力",
]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\w\u4e00-\u9fff]+", text.lower()))


def _similarity(a: str, b: str) -> float:
    left = _tokens(a)
    right = _tokens(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def detect_turn(turn: QATurn) -> AIGCResult:
    answer = turn.answer.strip()
    max_template = max(TEMPLATES, key=lambda template: _similarity(answer, template))
    template_similarity = _similarity(answer, max_template)
    polished_markers = ["首先", "其次", "最后", "综上", "显著提升", "业务痛点"]
    ai_generated_prob = min(
        1.0,
        0.15
        + 0.15 * sum(marker in answer for marker in polished_markers)
        + (0.25 if len(answer) > 180 and "我" not in answer[:80] else 0.0)
        + template_similarity * 0.4,
    )
    flagged = ai_generated_prob >= 0.65 or template_similarity >= 0.45
    return AIGCResult(
        turn_id=turn.turn_id,
        ai_generated_prob=round(ai_generated_prob, 3),
        template_similarity=round(template_similarity, 3),
        matched_template=max_template if template_similarity > 0.2 else None,
        flagged=flagged,
    )


def detect_interview(turns: list[QATurn]) -> list[AIGCResult]:
    return [detect_turn(turn) for turn in turns]

