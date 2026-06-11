from __future__ import annotations

from libs.schemas import (
    CandidateCreate,
    ChainLink,
    CompetencyItem,
    CompetencyModel,
    InterviewContext,
    InterviewCreate,
    JobCreate,
    ProbeChain,
    ProbeRequest,
    QATurn,
    TranscriptSegment,
    Utterance,
)
from libs.common.config import get_settings
from libs.common.database import init_db
from services.aigc_detect_service.service import detect_interview
from services.interview_orchestrator.dialogue import DialogueAssembler, FALLBACK_QUESTION
from services.interview_orchestrator.service import (
    add_turn,
    add_utterance,
    create_candidate,
    create_interview,
    get_interview,
    list_turns,
    list_utterances,
)
from services.jd_kb_service.service import create_job
from services.probe_service.service import fallback_probe
from services.scoring_service.service import fallback_score_interview


def _competencies() -> CompetencyModel:
    return CompetencyModel(
        job_id="job-1",
        job_title="AI 后端",
        items=[
            CompetencyItem(name="项目真实性", description="真实性", weight=0.6),
            CompetencyItem(name="注水风险", description="风险", weight=-0.1),
            CompetencyItem(name="沟通与逻辑", description="表达", weight=0.4),
        ],
    )


def _segment(
    speaker: str,
    text: str,
    start_ms: int,
    end_ms: int,
    *,
    session_id: str = "session-1",
) -> TranscriptSegment:
    return TranscriptSegment(
        session_id=session_id,
        speaker=speaker,  # type: ignore[arg-type]
        text=text,
        start_ms=start_ms,
        end_ms=end_ms,
        is_final=True,
        confidence=0.9,
    )


def test_dialogue_assembler_pairs_real_question_and_merged_answer() -> None:
    assembler = DialogueAssembler(silence_close_ms=2500)

    first = assembler.feed(_segment("interviewer", "请介绍核心项目。", 0, 900))
    assert first.turns == []
    assert first.utterances == []

    second = assembler.feed(_segment("candidate", "我负责网关。", 1200, 1800))
    assert second.turns == []
    assert second.utterances[0].speaker == "interviewer"

    assembler.feed(_segment("candidate", "主要做限流和降级。", 1900, 2600))
    sealed = assembler.feed(_segment("interviewer", "继续说指标。", 5200, 6000))

    assert sealed.utterances[0].speaker == "candidate"
    turn = sealed.turns[0]
    assert turn.question == "请介绍核心项目。"
    assert turn.answer == "我负责网关。 主要做限流和降级。"
    assert turn.question_utterance_id == second.utterances[0].utterance_id
    assert turn.answer_utterance_id == sealed.utterances[0].utterance_id


def test_dialogue_assembler_uses_explicit_fallback_for_candidate_first() -> None:
    assembler = DialogueAssembler()

    result = assembler.feed(
        _segment("candidate", "我先讲项目背景。", 0, 1000),
        force_close=True,
    )

    assert result.utterances[0].speaker == "candidate"
    assert result.turns[0].question == FALLBACK_QUESTION


def test_probe_chain_penalty_and_rehearsal_score_are_deterministic() -> None:
    first = QATurn(
        question="你具体负责哪一段？",
        answer="记不清了，主要是团队一起做的。",
        answer_start_ms=0,
        answer_end_ms=1000,
    )
    second = QATurn(
        question="那你写过哪段代码？",
        question_source="ai_probe",
        answer="这块主要也是团队负责，我参与了一些优化。",
        answer_start_ms=1500,
        answer_end_ms=2600,
        probe_target="简历写独立主导网关项目",
    )
    chain = ProbeChain(
        interview_id="session-1",
        topic="简历写独立主导网关项目",
        origin="resume_claim",
        links=[
            ChainLink(
                probe_question=first.question,
                probe_target="简历写独立主导网关项目",
                answer_turn_id=first.turn_id,
                credibility_after="suspicious",
            ),
            ChainLink(
                probe_question=second.question,
                probe_target="简历写独立主导网关项目",
                answer_turn_id=second.turn_id,
                credibility_after="vague",
            ),
        ],
        verdict="cracked",
        crack_depth=2,
    )
    ctx = InterviewContext(
        session_id="session-1",
        job_id="job-1",
        candidate_id="candidate-1",
        competency_model=_competencies(),
        turns=[first, second],
        probe_chains=[chain],
    )
    aigc = detect_interview(ctx.turns, probe_chains=ctx.probe_chains)
    score = fallback_score_interview(ctx, aigc)

    assert aigc[0].mode == "voice"
    assert aigc[0].rehearsal_score >= 0.0
    assert any("第 2 层追问露馅" in note for note in score.risk_notes)
    assert score.analysis_mode == "fallback"


def test_utterance_and_turn_refs_are_persisted(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'v2.db'}")
    get_settings.cache_clear()
    init_db()
    job = create_job(JobCreate(title="AI 后端", jd_text="FastAPI LLM"))
    candidate = create_candidate(
        CandidateCreate(name="候选人", resume_text="独立主导网关优化，响应时间提升 50%")
    )
    interview = create_interview(InterviewCreate(job_id=job.id, candidate_id=candidate.id))
    question = Utterance(
        speaker="interviewer",
        text="请介绍网关项目。",
        start_ms=0,
        end_ms=800,
        sentence_count=1,
    )
    answer = Utterance(
        speaker="candidate",
        text="我负责限流和降级。",
        start_ms=1000,
        end_ms=1800,
        sentence_count=1,
    )
    add_utterance(interview.id, question)
    add_utterance(interview.id, answer)
    add_turn(
        interview.id,
        QATurn(
            question=question.text,
            answer=answer.text,
            answer_start_ms=answer.start_ms,
            answer_end_ms=answer.end_ms,
            question_utterance_id=question.utterance_id,
            answer_utterance_id=answer.utterance_id,
        ),
    )

    assert [item.text for item in list_utterances(interview.id)] == [question.text, answer.text]
    persisted_turn = list_turns(interview.id)[0]
    assert persisted_turn.question_utterance_id == question.utterance_id
    assert persisted_turn.answer_utterance_id == answer.utterance_id


def test_probe_suggestion_chain_id_is_persisted_when_answered(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'chain.db'}")
    get_settings.cache_clear()
    init_db()
    job = create_job(JobCreate(title="AI 后端", jd_text="FastAPI LLM"))
    candidate = create_candidate(
        CandidateCreate(name="候选人", resume_text="独立主导网关优化，响应时间提升 50%")
    )
    interview = create_interview(InterviewCreate(job_id=job.id, candidate_id=candidate.id))
    chain = interview.context.probe_chains[0]
    response = fallback_probe(
        ProbeRequest(
            job_id=job.id,
            competency_model=job.competency_model,
            latest_answer="我主要负责优化，效果比较明显。",
            recent_turns=[],
            probe_chains=interview.context.probe_chains,
        )
    )
    suggestion = response.suggestions[0]

    assert suggestion.chain_id == chain.chain_id
    assert suggestion.chain_label

    add_turn(
        interview.id,
        QATurn(
            question=suggestion.question,
            question_source="ai_probe",
            answer="记不清了，主要是团队一起做的。",
            answer_start_ms=0,
            answer_end_ms=1000,
            probe_target=suggestion.target,
            probe_chain_id=suggestion.chain_id,
        ),
    )
    persisted = get_interview(interview.id)
    persisted_chain = persisted.context.probe_chains[0]

    assert persisted.context.turns[0].probe_chain_id == chain.chain_id
    assert persisted_chain.links[0].answer_turn_id == persisted.context.turns[0].turn_id
    assert persisted_chain.links[0].credibility_after == "suspicious"
