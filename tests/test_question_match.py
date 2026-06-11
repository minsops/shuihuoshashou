from __future__ import annotations

from libs.common.config import get_settings
from libs.schemas import NextOption, NextOptions, QuestionBank, QuestionBankItem
from services.interview_orchestrator.question_match import match_asked_question


def _bank(*, asked_first: bool = False) -> QuestionBank:
    questions = [
        (
            "bank-1",
            "technical",
            "请讲你在网关系统里亲自写了哪段代码？",
            asked_first,
        ),
        (
            "bank-2",
            "project",
            "线上故障你如何定位根因并完成恢复？",
            False,
        ),
        ("bank-3", "experience", "你在团队协作里承担了什么角色？", False),
        ("bank-4", "job_match", "为什么你适合这个 AI 后端岗位？", False),
        ("bank-5", "behavior", "遇到需求变化时你如何推进沟通？", False),
        ("bank-6", "technical", "限流模块的核心指标怎么设计？", False),
        ("bank-7", "project", "项目上线后你如何复盘效果？", False),
        ("bank-8", "experience", "请举例说明一次跨团队协作。", False),
    ]
    return QuestionBank(
        interview_id="interview-1",
        items=[
            QuestionBankItem(
                question_id=question_id,
                category=category,  # type: ignore[arg-type]
                question=question,
                basis="resume",
                basis_excerpt="简历提到相关经历",
                competency="项目真实性",
                asked=asked,
            )
            for question_id, category, question, asked in questions
        ],
    )


def test_match_asked_question_matches_current_question_exactly(monkeypatch) -> None:
    monkeypatch.setenv("QUESTION_MATCH_THRESHOLD", "0.30")
    get_settings.cache_clear()
    bank = _bank()
    current = NextOption(
        option_id="option-current",
        kind="bank",
        question="请讲你在网关系统里亲自写了哪段代码？",
        reason="验证是否真实参与实现",
        bank_question_id="bank-1",
    )

    matched_id, stored_question = match_asked_question(current.question, current, None, bank)

    assert matched_id == "option-current"
    assert stored_question == current.question
    assert bank.items[0].asked is True


def test_match_asked_question_matches_paraphrased_option(monkeypatch) -> None:
    monkeypatch.setenv("QUESTION_MATCH_THRESHOLD", "0.30")
    get_settings.cache_clear()
    bank = _bank()
    follow_up = NextOption(
        option_id="option-follow",
        kind="follow_up",
        question="请讲你在网关系统里亲自写了哪段代码？",
        reason="回答含糊，继续追问亲历细节",
        bank_question_id="bank-1",
    )
    options = NextOptions(
        interview_id="interview-1",
        after_turn_id="turn-1",
        follow_up=[follow_up],
        alternatives=[
            NextOption(
                option_id="option-alt-1",
                kind="bank",
                question="为什么你适合这个 AI 后端岗位？",
                reason="考察岗位匹配度",
            ),
            NextOption(
                option_id="option-alt-2",
                kind="generated",
                question="你在团队协作里承担了什么角色？",
                reason="考察协作真实性",
            ),
        ],
    )

    matched_id, stored_question = match_asked_question(
        "网关系统你亲自写了哪段代码",
        None,
        options,
        bank,
    )

    assert matched_id == "option-follow"
    assert stored_question == follow_up.question
    assert bank.items[0].asked is True


def test_match_asked_question_keeps_custom_question_original(monkeypatch) -> None:
    monkeypatch.setenv("QUESTION_MATCH_THRESHOLD", "0.30")
    get_settings.cache_clear()
    bank = _bank()

    matched_id, stored_question = match_asked_question(
        "你今天午饭吃了什么？",
        None,
        None,
        bank,
    )

    assert matched_id is None
    assert stored_question == "你今天午饭吃了什么？"
    assert all(item.asked is False for item in bank.items)


def test_match_asked_question_uses_unasked_bank_items_and_ignores_asked_items(monkeypatch) -> None:
    monkeypatch.setenv("QUESTION_MATCH_THRESHOLD", "0.30")
    get_settings.cache_clear()
    bank = _bank(asked_first=True)

    matched_id, stored_question = match_asked_question(
        "网关系统你亲自写了哪段代码",
        None,
        None,
        bank,
    )

    assert matched_id is None
    assert stored_question == "网关系统你亲自写了哪段代码"
    assert bank.items[0].asked is True

    matched_id, stored_question = match_asked_question(
        "线上故障你怎么定位和恢复的",
        None,
        None,
        bank,
    )

    assert matched_id == "bank-2"
    assert stored_question == "线上故障你如何定位根因并完成恢复？"
    assert bank.items[1].asked is True
