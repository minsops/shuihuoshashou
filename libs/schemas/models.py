from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


def new_id() -> str:
    return str(uuid4())


class InterviewStatus(str, Enum):
    created = "CREATED"
    in_progress = "IN_PROGRESS"
    finished = "FINISHED"
    scoring = "SCORING"
    reported = "REPORTED"


class CompetencyItem(BaseModel):
    name: str
    description: str
    probe_patterns: list[str] = Field(default_factory=list)
    weight: float = 0.0


class CompetencyModel(BaseModel):
    job_id: str
    job_title: str
    items: list[CompetencyItem]

    @field_validator("items")
    @classmethod
    def has_items(cls, value: list[CompetencyItem]) -> list[CompetencyItem]:
        if not value:
            raise ValueError("competency model must include at least one item")
        return value


class ProbePatternHit(BaseModel):
    job_id: str
    competency: str
    pattern: str
    score: float = Field(ge=0.0)


class TranscriptSegment(BaseModel):
    session_id: str
    speaker: Literal["interviewer", "candidate", "unknown"]
    text: str
    start_ms: int
    end_ms: int
    is_final: bool
    confidence: float = Field(ge=0.0, le=1.0)


class ConsistencyFlag(BaseModel):
    turn_id_a: str
    turn_id_b: str
    description: str
    severity: Literal["low", "high"]


class FactClaim(BaseModel):
    turn_id: str
    contribution_scope: Literal["solo", "lead", "participant", "team", "unknown"] = "unknown"
    responsibilities: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)


class QATurn(BaseModel):
    turn_id: str = Field(default_factory=new_id)
    question: str
    question_source: Literal["interviewer", "ai_probe"] = "interviewer"
    answer: str
    answer_start_ms: int = 0
    answer_end_ms: int = 0
    probe_target: str | None = None


class InterviewContext(BaseModel):
    session_id: str
    job_id: str
    candidate_id: str
    competency_model: CompetencyModel
    turns: list[QATurn] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ended_at: datetime | None = None
    flags: list[ConsistencyFlag] = Field(default_factory=list)


class ProbeRequest(BaseModel):
    job_id: str
    competency_model: CompetencyModel
    recent_turns: list[QATurn]
    latest_answer: str


class ProbeSuggestion(BaseModel):
    question: str
    target: str
    competency: str
    priority: int = Field(ge=1, le=3)


class CredibilitySignal(BaseModel):
    level: Literal["solid", "vague", "suspicious"]
    reason: str
    drill_down_hint: str


class ProbeResponse(BaseModel):
    suggestions: list[ProbeSuggestion]
    credibility: CredibilitySignal


class EvidenceRef(BaseModel):
    turn_id: str
    quote_start_ms: int
    quote_end_ms: int
    excerpt: str


class DimensionScore(BaseModel):
    dimension: str
    score: float = Field(ge=0.0, le=100.0)
    weight: float
    evidence: list[EvidenceRef] = Field(default_factory=list)


class InterviewScore(BaseModel):
    session_id: str
    dimensions: list[DimensionScore]
    total_score: float = Field(ge=0.0, le=100.0)
    risk_notes: list[str] = Field(default_factory=list)
    recommendation: Literal["strong_yes", "yes", "hold", "no"]


class AIGCResult(BaseModel):
    turn_id: str
    ai_generated_prob: float = Field(ge=0.0, le=1.0)
    template_similarity: float = Field(ge=0.0, le=1.0)
    matched_template: str | None = None
    flagged: bool = False


class BehaviorSignal(BaseModel):
    turn_id: str
    fluency: float = Field(ge=0.0, le=1.0)
    hesitation: float = Field(ge=0.0, le=1.0)
    evasiveness_hint: bool


class JobCreate(BaseModel):
    title: str
    jd_text: str


class JobRecord(BaseModel):
    id: str = Field(default_factory=new_id)
    title: str
    jd_text: str
    competency_model: CompetencyModel
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CandidateCreate(BaseModel):
    name: str
    resume_text: str = ""


class CandidateRecord(BaseModel):
    id: str = Field(default_factory=new_id)
    name: str
    resume_text: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ConsentCreate(BaseModel):
    candidate_id: str
    consent_type: Literal["behavior_signal"] = "behavior_signal"
    granted: bool = True


class ConsentRecord(BaseModel):
    id: str = Field(default_factory=new_id)
    candidate_id: str
    consent_type: Literal["behavior_signal"] = "behavior_signal"
    granted: bool = True
    granted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    revoked_at: datetime | None = None


class InterviewCreate(BaseModel):
    job_id: str
    candidate_id: str
    signal_enabled: bool = False


class InterviewRecord(BaseModel):
    id: str = Field(default_factory=new_id)
    job_id: str
    candidate_id: str
    status: InterviewStatus = InterviewStatus.created
    context: InterviewContext
    signal_enabled: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    ended_at: datetime | None = None


class OfflineInterviewInput(BaseModel):
    job_title: str
    jd_text: str
    candidate_name: str
    resume_text: str = ""
    turns: list[QATurn]


class OfflineInterviewResult(BaseModel):
    job: JobRecord
    candidate: CandidateRecord
    interview: InterviewRecord
    report: "Report"


class Report(BaseModel):
    interview_id: str
    score: InterviewScore
    aigc_results: list[AIGCResult]
    consistency_flags: list[ConsistencyFlag]
    summary: str
    html_path: str | None = None
    pdf_path: str | None = None
    artifact_uris: dict[str, str] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
