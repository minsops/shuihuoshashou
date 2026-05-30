from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from libs.schemas import AIGCResult, QATurn


TEMPLATE_PATH = Path(__file__).with_name("templates") / "common_answer_templates.txt"


@lru_cache
def load_templates() -> tuple[str, ...]:
    return tuple(
        line.strip()
        for line in TEMPLATE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    )


def _normalize(text: str) -> str:
    return "".join(re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", text.lower()))


def _char_ngrams(text: str, size: int = 2) -> Counter[str]:
    normalized = _normalize(text)
    if not normalized:
        return Counter()
    if len(normalized) <= size:
        return Counter([normalized])
    return Counter(normalized[index : index + size] for index in range(len(normalized) - size + 1))


def _cosine_similarity(a: str, b: str) -> float:
    left = _char_ngrams(a)
    right = _char_ngrams(b)
    if not left or not right:
        return 0.0
    overlap = sum(left[token] * right[token] for token in left.keys() & right.keys())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return overlap / (left_norm * right_norm)


def detect_turn(turn: QATurn) -> AIGCResult:
    answer = turn.answer.strip()
    templates = load_templates()
    max_template = max(templates, key=lambda template: _cosine_similarity(answer, template))
    template_similarity = _cosine_similarity(answer, max_template)
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
