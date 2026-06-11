from __future__ import annotations

from dataclasses import dataclass

from libs.schemas import NextOption, NextOptions, QATurn, QuestionBank, QuestionBankItem
from services.probe_service.service import assess_credibility


@dataclass
class QuestionFlowState:
    bank: QuestionBank
    current_question: NextOption | None = None
    last_options: NextOptions | None = None


def initial_question(bank: QuestionBank) -> NextOption | None:
    item = _preferred_bank_item(bank.items)
    if item is None:
        return None
    return _bank_option(item, reason="优先验证简历与岗位要求中最关键的信息。")


def fallback_next_options(record, bank: QuestionBank, after_turn: QATurn) -> NextOptions:
    follow_up = _follow_up_option(record, after_turn)
    alternatives = [_bank_option(item, reason="题库中尚未提问，适合切换话题。") for item in _unasked_items(bank)]
    alternatives = alternatives[:3]
    while len(alternatives) < 2:
        alternatives.append(_generated_alternative(record, len(alternatives) + 1))
    return NextOptions(
        interview_id=bank.interview_id,
        after_turn_id=after_turn.turn_id,
        follow_up=[follow_up],
        alternatives=alternatives,
    )


def find_option(flow: QuestionFlowState, option_id: str) -> NextOption | None:
    if flow.current_question is not None and flow.current_question.option_id == option_id:
        return flow.current_question
    if flow.last_options is None:
        return None
    for option in [*flow.last_options.follow_up, *flow.last_options.alternatives]:
        if option.option_id == option_id:
            return option
    return None


def _preferred_bank_item(items: list[QuestionBankItem]) -> QuestionBankItem | None:
    unasked = [item for item in items if not item.asked]
    for item in unasked:
        if item.basis in {"resume", "jd_resume"} and item.category == "project":
            return item
    return unasked[0] if unasked else None


def _unasked_items(bank: QuestionBank) -> list[QuestionBankItem]:
    return [item for item in bank.items if not item.asked]


def _bank_option(item: QuestionBankItem, *, reason: str) -> NextOption:
    return NextOption(
        kind="bank",
        question=item.question,
        reason=reason,
        category=item.category,
        bank_question_id=item.question_id,
    )


def _follow_up_option(record, after_turn: QATurn) -> NextOption:
    credibility = assess_credibility(after_turn.answer)
    chain = record.context.probe_chains[-1] if record.context.probe_chains else None
    if credibility.level in {"suspicious", "vague"}:
        question = (
            "刚才这段回答还比较概括，请讲一个具体例子：你本人负责哪一段，"
            "关键指标是什么，遇到异常时怎么处理？"
        )
    else:
        question = (
            "沿着刚才的回答，请补充一个边界场景：你如何验证方案有效，"
            "失败或效果不达标时怎么复盘？"
        )
    return NextOption(
        kind="follow_up",
        question=question,
        reason=credibility.reason,
        chain_id=chain.chain_id if chain is not None else after_turn.probe_chain_id,
    )


def _generated_alternative(record, index: int) -> NextOption:
    competencies = record.context.competency_model.items
    competency = competencies[(index - 1) % len(competencies)].name if competencies else "项目真实性"
    return NextOption(
        kind="generated",
        question=f"围绕「{competency}」，请讲一个可以验证你个人贡献的具体案例。",
        reason="题库可用问题不足，自动补齐备选。",
    )
