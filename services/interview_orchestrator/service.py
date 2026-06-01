from __future__ import annotations

from datetime import UTC, datetime

from libs.common.config import get_settings
from libs.common.database import connect, dumps, init_db, loads
from libs.common.events import event_bus
from libs.common.tasks import task_queue
from libs.schemas import (
    CandidateCreate,
    CandidateRecord,
    ConsentCreate,
    ConsentRecord,
    InterviewContext,
    InterviewCreate,
    InterviewRecord,
    InterviewStatus,
    OfflineTaskAccepted,
    QATurn,
    TranscriptSegment,
)
from services.aigc_detect_service.service import detect_interview
from services.interview_orchestrator.consistency import detect_claim_conflicts, extract_fact_claim
from services.jd_kb_service.service import get_job
from services.report_service.service import build_report
from services.scoring_service.service import score_interview


def create_candidate(payload: CandidateCreate) -> CandidateRecord:
    init_db()
    record = CandidateRecord(name=payload.name, resume_text=payload.resume_text)
    with connect() as conn:
        conn.execute(
            "INSERT INTO candidates (id, name, resume_text, created_at) VALUES (?, ?, ?, ?)",
            (record.id, record.name, record.resume_text, record.created_at.isoformat()),
        )
    return record


def create_consent(payload: ConsentCreate) -> ConsentRecord:
    init_db()
    now = datetime.now(UTC)
    if not payload.granted:
        with connect() as conn:
            conn.execute(
                """
                UPDATE consents
                SET revoked_at = ?
                WHERE candidate_id = ?
                  AND consent_type = ?
                  AND granted = 1
                  AND revoked_at IS NULL
                """,
                (now.isoformat(), payload.candidate_id, payload.consent_type),
            )
    record = ConsentRecord(
        candidate_id=payload.candidate_id,
        consent_type=payload.consent_type,
        granted=payload.granted,
        granted_at=now,
        revoked_at=now if not payload.granted else None,
    )
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO consents
            (id, candidate_id, consent_type, granted, granted_at, revoked_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.candidate_id,
                record.consent_type,
                record.granted,
                record.granted_at.isoformat(),
                record.revoked_at.isoformat() if record.revoked_at else None,
            ),
        )
    return record


def has_active_consent(candidate_id: str, consent_type: str) -> bool:
    init_db()
    with connect() as conn:
        row = conn.execute(
            """
            SELECT 1 FROM consents
            WHERE candidate_id = ?
              AND consent_type = ?
              AND granted = 1
              AND revoked_at IS NULL
            LIMIT 1
            """,
            (candidate_id, consent_type),
        ).fetchone()
    return row is not None


def create_interview(payload: InterviewCreate) -> InterviewRecord:
    init_db()
    if payload.signal_enabled:
        if not get_settings().signal_enabled:
            raise PermissionError("behavior signal requires admin enablement")
        if not has_active_consent(payload.candidate_id, "behavior_signal"):
            raise PermissionError("behavior signal requires explicit candidate consent")
    job = get_job(payload.job_id)
    ctx = InterviewContext(
        session_id="",
        job_id=payload.job_id,
        candidate_id=payload.candidate_id,
        competency_model=job.competency_model,
    )
    record = InterviewRecord(
        job_id=payload.job_id,
        candidate_id=payload.candidate_id,
        context=ctx,
        signal_enabled=payload.signal_enabled,
    )
    record.context.session_id = record.id
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO interviews
            (id, job_id, candidate_id, status, context, signal_enabled, created_at, started_at, ended_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.job_id,
                record.candidate_id,
                record.status.value,
                dumps(record.context.model_dump()),
                record.signal_enabled,
                record.created_at.isoformat(),
                None,
                None,
            ),
        )
    return record


def get_interview(interview_id: str) -> InterviewRecord:
    init_db()
    with connect() as conn:
        row = conn.execute("SELECT * FROM interviews WHERE id = ?", (interview_id,)).fetchone()
    if row is None:
        raise KeyError(f"interview not found: {interview_id}")
    return InterviewRecord(
        id=row["id"],
        job_id=row["job_id"],
        candidate_id=row["candidate_id"],
        status=InterviewStatus(row["status"]),
        context=InterviewContext.model_validate(loads(row["context"])),
        signal_enabled=bool(row["signal_enabled"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
        ended_at=datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None,
    )


def save_interview(record: InterviewRecord) -> None:
    with connect() as conn:
        conn.execute(
            """
            UPDATE interviews
            SET status = ?, context = ?, signal_enabled = ?, started_at = ?, ended_at = ?
            WHERE id = ?
            """,
            (
                record.status.value,
                dumps(record.context.model_dump()),
                record.signal_enabled,
                record.started_at.isoformat() if record.started_at else None,
                record.ended_at.isoformat() if record.ended_at else None,
                record.id,
            ),
        )


def start_interview(interview_id: str) -> InterviewRecord:
    record = get_interview(interview_id)
    if record.status == InterviewStatus.in_progress:
        return record
    if record.status != InterviewStatus.created:
        raise ValueError(f"cannot start interview from status {record.status.value}")
    record.status = InterviewStatus.in_progress
    record.started_at = datetime.now(UTC)
    record.context.started_at = record.started_at
    save_interview(record)
    return record


def add_turn(interview_id: str, turn: QATurn) -> InterviewRecord:
    record = get_interview(interview_id)
    if record.status not in {InterviewStatus.created, InterviewStatus.in_progress}:
        raise ValueError(f"cannot add turn to interview in status {record.status.value}")
    record.context.turns.append(turn)
    if len(record.context.fact_claims) != len(record.context.turns) - 1:
        record.context.fact_claims = [
            extract_fact_claim(existing_turn) for existing_turn in record.context.turns[:-1]
        ]
    record.context.fact_claims.append(extract_fact_claim(turn))
    record.context.flags = detect_claim_conflicts(record.context.fact_claims)
    save_interview(record)
    with connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO qa_turns
            (id, interview_id, turn_index, question, question_source, answer,
             answer_start_ms, answer_end_ms, probe_target, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                turn.turn_id,
                interview_id,
                len(record.context.turns) - 1,
                turn.question,
                turn.question_source,
                turn.answer,
                turn.answer_start_ms,
                turn.answer_end_ms,
                turn.probe_target,
                dumps(turn.model_dump()),
            ),
        )
    event_bus.publish_nowait(
        "qa_turn.created",
        {
            "interview_id": interview_id,
            "turn_id": turn.turn_id,
            "turn_index": len(record.context.turns) - 1,
        },
    )
    return record


def list_turns(interview_id: str) -> list[QATurn]:
    init_db()
    with connect() as conn:
        rows = conn.execute(
            "SELECT payload FROM qa_turns WHERE interview_id = ? ORDER BY turn_index",
            (interview_id,),
        ).fetchall()
    return [QATurn.model_validate(loads(row["payload"])) for row in rows]


def should_probe(segment: TranscriptSegment, record: InterviewRecord) -> bool:
    settings = get_settings()
    if not segment.is_final or segment.speaker != "candidate":
        return False
    if len(segment.text.strip()) < settings.probe_min_answer_chars:
        return False
    if settings.probe_require_topic_match and not _is_drill_down_topic(segment.text):
        return False
    if not record.context.turns:
        return True
    last_turn = record.context.turns[-1]
    return segment.start_ms - last_turn.answer_end_ms >= settings.probe_min_interval_ms


def _is_drill_down_topic(text: str) -> bool:
    normalized = text.strip().lower()
    keywords = [
        item.strip().lower()
        for item in get_settings().probe_topic_keywords.split(",")
        if item.strip()
    ]
    return any(keyword in normalized for keyword in keywords)


def finish_interview(interview_id: str) -> InterviewRecord:
    record = get_interview(interview_id)
    if record.status == InterviewStatus.finished:
        return record
    if record.status not in {InterviewStatus.created, InterviewStatus.in_progress}:
        raise ValueError(f"cannot finish interview from status {record.status.value}")
    record.status = InterviewStatus.finished
    record.ended_at = datetime.now(UTC)
    record.context.ended_at = record.ended_at
    save_interview(record)
    event_bus.publish_nowait(
        "interview.finished",
        {"interview_id": interview_id, "ended_at": record.ended_at.isoformat()},
    )
    return record


def run_offline_scoring_task(interview_id: str):
    record = get_interview(interview_id)
    if record.status != InterviewStatus.finished:
        raise ValueError(f"offline scoring requires FINISHED status, got {record.status.value}")
    record.status = InterviewStatus.scoring
    save_interview(record)
    event_bus.publish_nowait(
        "interview.scoring_started",
        {"interview_id": interview_id, "turn_count": len(record.context.turns)},
    )
    aigc = detect_interview(record.context.turns)
    score = score_interview(record.context, aigc)
    report, html = build_report(record.context, score, aigc)
    record.status = InterviewStatus.reported
    save_interview(record)
    with connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO scores
            (interview_id, dimensions, total_score, risk_notes, recommendation, payload)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                interview_id,
                dumps([dimension.model_dump() for dimension in score.dimensions]),
                score.total_score,
                dumps(score.risk_notes),
                score.recommendation,
                dumps(score.model_dump()),
            ),
        )
        for item in aigc:
            conn.execute(
                """
                INSERT OR REPLACE INTO aigc_results
                (id, interview_id, turn_id, ai_generated_prob, template_similarity,
                 matched_template, flagged, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.turn_id,
                    interview_id,
                    item.turn_id,
                    item.ai_generated_prob,
                    item.template_similarity,
                    item.matched_template,
                    item.flagged,
                    dumps(item.model_dump()),
                ),
            )
        conn.execute(
            "INSERT OR REPLACE INTO reports (interview_id, payload, html) VALUES (?, ?, ?)",
            (interview_id, dumps(report.model_dump()), html),
        )
    event_bus.publish_nowait(
        "interview.reported",
        {
            "interview_id": interview_id,
            "total_score": report.score.total_score,
            "recommendation": report.score.recommendation,
        },
    )
    return report


def enqueue_offline_scoring_task(interview_id: str, *, execute_inline: bool | None = None):
    if execute_inline is None:
        execute_inline = get_settings().offline_task_execution != "async"
    if not execute_inline:
        record = task_queue.enqueue_deferred(
            "interview.offline_scoring",
            {"interview_id": interview_id},
        )
        return OfflineTaskAccepted(
            interview_id=interview_id,
            task_id=record.task_id,
            task_name=record.name,
            status="queued",
        )
    record = task_queue.enqueue(
        "interview.offline_scoring",
        {"interview_id": interview_id},
        lambda payload: run_offline_scoring_task(str(payload["interview_id"])),
    )
    return record.result


def end_interview(interview_id: str, *, execute_inline: bool | None = None):
    finish_interview(interview_id)
    return enqueue_offline_scoring_task(interview_id, execute_inline=execute_inline)


def get_report(interview_id: str):
    init_db()
    with connect() as conn:
        row = conn.execute("SELECT payload, html FROM reports WHERE interview_id = ?", (interview_id,)).fetchone()
    if row is None:
        raise KeyError(f"report not found: {interview_id}")
    return loads(row["payload"]), row["html"]
