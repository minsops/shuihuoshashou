from __future__ import annotations

from dataclasses import dataclass

from libs.common.config import get_settings
from libs.common.textsim import cosine_similarity, normalize_text
from libs.schemas import NextOption, NextOptions, QuestionBank


@dataclass(frozen=True)
class _QuestionCandidate:
    option_id: str
    question: str
    bank_question_id: str | None = None


def match_asked_question(
    utterance_text: str,
    current: NextOption | None,
    options: NextOptions | None,
    bank: QuestionBank,
) -> tuple[str | None, str]:
    asked_text = utterance_text.strip()
    candidates = _collect_candidates(current, options, bank)
    if not asked_text or not candidates:
        return None, asked_text

    best = max(candidates, key=lambda candidate: cosine_similarity(asked_text, candidate.question))
    score = cosine_similarity(asked_text, best.question)
    if score < get_settings().question_match_threshold:
        return None, asked_text

    _mark_bank_item_asked(bank, best.bank_question_id)
    return best.option_id, best.question


def _collect_candidates(
    current: NextOption | None,
    options: NextOptions | None,
    bank: QuestionBank,
) -> list[_QuestionCandidate]:
    candidates: list[_QuestionCandidate] = []
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()

    def add(option_id: str, question: str, bank_question_id: str | None = None) -> None:
        normalized_question = normalize_text(question)
        if option_id in seen_ids or normalized_question in seen_questions:
            return
        candidates.append(_QuestionCandidate(option_id, question, bank_question_id))
        seen_ids.add(option_id)
        seen_questions.add(normalized_question)

    if current is not None:
        add(current.option_id, current.question, current.bank_question_id)
    if options is not None:
        for option in [*options.follow_up, *options.alternatives]:
            add(option.option_id, option.question, option.bank_question_id)
    for item in bank.items:
        if not item.asked:
            add(item.question_id, item.question, item.question_id)
    return candidates


def _mark_bank_item_asked(bank: QuestionBank, bank_question_id: str | None) -> None:
    if bank_question_id is None:
        return
    for item in bank.items:
        if item.question_id == bank_question_id:
            item.asked = True
            return
