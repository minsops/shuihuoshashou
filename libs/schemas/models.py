from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator


def new_id() -> str:
    return str(uuid4())


def _not_blank(value: str, label: str) -> str:
    if not value.strip():
        raise ValueError(f"{label} must not be blank")
    return value


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

    @field_validator("name", "description")
    @classmethod
    def text_fields_are_not_blank(cls, value: str) -> str:
        return _not_blank(value, "competency text")

    @field_validator("probe_patterns")
    @classmethod
    def probe_patterns_are_not_blank(cls, value: list[str]) -> list[str]:
        if any(not pattern.strip() for pattern in value):
            raise ValueError("probe patterns must not contain blank text")
        return value


class CompetencyModel(BaseModel):
    job_id: str
    job_title: str
    items: list[CompetencyItem]

    @field_validator("job_id")
    @classmethod
    def job_id_is_not_blank(cls, value: str) -> str:
        return _not_blank(value, "job_id")

    @field_validator("job_title")
    @classmethod
    def job_title_is_not_blank(cls, value: str) -> str:
        return _not_blank(value, "job_title")

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

    @field_validator("job_id")
    @classmethod
    def job_id_is_not_blank(cls, value: str) -> str:
        return _not_blank(value, "job_id")

    @field_validator("competency", "pattern")
    @classmethod
    def text_fields_are_not_blank(cls, value: str) -> str:
        return _not_blank(value, "probe pattern hit text")


class TranscriptSegment(BaseModel):
    session_id: str
    speaker: Literal["interviewer", "candidate", "unknown"]
    text: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    is_final: bool
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def timestamps_are_monotonic(self) -> "TranscriptSegment":
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must be greater than or equal to start_ms")
        return self

    @field_validator("session_id", "text")
    @classmethod
    def required_text_fields_are_not_blank(cls, value: str, info: ValidationInfo) -> str:
        return _not_blank(value, f"transcript segment {info.field_name}")


class ConsistencyFlag(BaseModel):
    turn_id_a: str
    turn_id_b: str
    description: str
    severity: Literal["low", "high"]

    @field_validator("turn_id_a", "turn_id_b")
    @classmethod
    def turn_ids_are_not_blank(cls, value: str, info: ValidationInfo) -> str:
        return _not_blank(value, info.field_name)

    @field_validator("description")
    @classmethod
    def description_is_not_blank(cls, value: str) -> str:
        return _not_blank(value, "description")


class FactClaim(BaseModel):
    turn_id: str
    contribution_scope: Literal["solo", "lead", "participant", "team", "unknown"] = "unknown"
    responsibilities: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)

    @field_validator("turn_id")
    @classmethod
    def turn_id_is_not_blank(cls, value: str) -> str:
        return _not_blank(value, "turn_id")


class QATurn(BaseModel):
    turn_id: str = Field(default_factory=new_id)
    question: str
    question_source: Literal["interviewer", "ai_probe"] = "interviewer"
    answer: str
    answer_start_ms: int = Field(default=0, ge=0)
    answer_end_ms: int = Field(default=0, ge=0)
    probe_target: str | None = None

    @model_validator(mode="after")
    def answer_timestamps_are_monotonic(self) -> "QATurn":
        if self.answer_end_ms < self.answer_start_ms:
            raise ValueError("answer_end_ms must be greater than or equal to answer_start_ms")
        return self

    @field_validator("turn_id")
    @classmethod
    def turn_id_is_not_blank(cls, value: str) -> str:
        return _not_blank(value, "turn_id")

    @field_validator("question", "answer")
    @classmethod
    def text_fields_are_not_blank(cls, value: str) -> str:
        return _not_blank(value, "question and answer")


class InterviewContext(BaseModel):
    session_id: str
    job_id: str
    candidate_id: str
    competency_model: CompetencyModel
    turns: list[QATurn] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ended_at: datetime | None = None
    fact_claims: list[FactClaim] = Field(default_factory=list)
    flags: list[ConsistencyFlag] = Field(default_factory=list)

    @field_validator("session_id", "job_id", "candidate_id")
    @classmethod
    def identifiers_are_not_blank(cls, value: str, info: ValidationInfo) -> str:
        return _not_blank(value, f"interview context {info.field_name}")


class ProbeRequest(BaseModel):
    job_id: str
    competency_model: CompetencyModel
    recent_turns: list[QATurn]
    latest_answer: str

    @field_validator("job_id")
    @classmethod
    def job_id_is_not_blank(cls, value: str) -> str:
        return _not_blank(value, "job_id")

    @field_validator("latest_answer")
    @classmethod
    def latest_answer_is_not_blank(cls, value: str) -> str:
        return _not_blank(value, "latest_answer")


class ProbeSuggestion(BaseModel):
    question: str
    target: str
    competency: str
    priority: int = Field(ge=1, le=3)

    @field_validator("question", "target", "competency")
    @classmethod
    def text_fields_are_not_blank(cls, value: str) -> str:
        return _not_blank(value, "probe suggestion text")


class CredibilitySignal(BaseModel):
    level: Literal["solid", "vague", "suspicious"]
    reason: str
    drill_down_hint: str

    @field_validator("reason", "drill_down_hint")
    @classmethod
    def text_fields_are_not_blank(cls, value: str) -> str:
        return _not_blank(value, "credibility text")


class ProbeResponse(BaseModel):
    suggestions: list[ProbeSuggestion] = Field(min_length=1, max_length=3)
    credibility: CredibilitySignal


class EvidenceRef(BaseModel):
    turn_id: str
    quote_start_ms: int = Field(ge=0)
    quote_end_ms: int = Field(ge=0)
    excerpt: str

    @model_validator(mode="after")
    def quote_timestamps_are_monotonic(self) -> "EvidenceRef":
        if self.quote_end_ms < self.quote_start_ms:
            raise ValueError("quote_end_ms must be greater than or equal to quote_start_ms")
        return self

    @field_validator("turn_id")
    @classmethod
    def turn_id_is_not_blank(cls, value: str) -> str:
        return _not_blank(value, "turn_id")

    @field_validator("excerpt")
    @classmethod
    def excerpt_is_not_blank(cls, value: str) -> str:
        return _not_blank(value, "excerpt")


class DimensionScore(BaseModel):
    dimension: str
    score: float = Field(ge=0.0, le=100.0)
    weight: float
    evidence: list[EvidenceRef] = Field(min_length=1)

    @field_validator("dimension")
    @classmethod
    def dimension_is_not_blank(cls, value: str) -> str:
        return _not_blank(value, "dimension")


class InterviewScore(BaseModel):
    session_id: str
    dimensions: list[DimensionScore] = Field(min_length=1)
    total_score: float = Field(ge=0.0, le=100.0)
    risk_notes: list[str] = Field(default_factory=list)
    recommendation: Literal["strong_yes", "yes", "hold", "no"]

    @field_validator("session_id")
    @classmethod
    def session_id_is_not_blank(cls, value: str) -> str:
        return _not_blank(value, "session_id")

    @field_validator("risk_notes")
    @classmethod
    def risk_notes_are_not_blank(cls, value: list[str]) -> list[str]:
        for note in value:
            _not_blank(note, "risk note")
        return value


class AIGCResult(BaseModel):
    turn_id: str
    ai_generated_prob: float = Field(ge=0.0, le=1.0)
    template_similarity: float = Field(ge=0.0, le=1.0)
    matched_template: str | None = None
    flagged: bool = False

    @field_validator("turn_id")
    @classmethod
    def turn_id_is_not_blank(cls, value: str) -> str:
        return _not_blank(value, "turn_id")


class BehaviorSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn_id: str
    fluency: float = Field(ge=0.0, le=1.0)
    hesitation: float = Field(ge=0.0, le=1.0)
    evasiveness_hint: bool

    @field_validator("turn_id")
    @classmethod
    def turn_id_is_not_blank(cls, value: str) -> str:
        return _not_blank(value, "turn_id")


class JobCreate(BaseModel):
    title: str
    jd_text: str

    @field_validator("title", "jd_text")
    @classmethod
    def text_fields_are_not_blank(cls, value: str) -> str:
        return _not_blank(value, "job text")


class JobRecord(BaseModel):
    id: str = Field(default_factory=new_id)
    title: str
    jd_text: str
    competency_model: CompetencyModel
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("title", "jd_text")
    @classmethod
    def text_fields_are_not_blank(cls, value: str) -> str:
        return _not_blank(value, "job text")

    @field_validator("id")
    @classmethod
    def id_is_not_blank(cls, value: str) -> str:
        return _not_blank(value, "job id")


class CandidateCreate(BaseModel):
    name: str
    resume_text: str = ""

    @field_validator("name")
    @classmethod
    def name_is_not_blank(cls, value: str) -> str:
        return _not_blank(value, "candidate name")


class CandidateRecord(BaseModel):
    id: str = Field(default_factory=new_id)
    name: str
    resume_text: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("name")
    @classmethod
    def name_is_not_blank(cls, value: str) -> str:
        return _not_blank(value, "candidate name")

    @field_validator("id")
    @classmethod
    def id_is_not_blank(cls, value: str) -> str:
        return _not_blank(value, "candidate id")


class ConsentCreate(BaseModel):
    candidate_id: str
    consent_type: Literal["behavior_signal"] = "behavior_signal"
    granted: bool = True

    @field_validator("candidate_id")
    @classmethod
    def candidate_id_is_not_blank(cls, value: str) -> str:
        return _not_blank(value, "candidate_id")


class ConsentRecord(BaseModel):
    id: str = Field(default_factory=new_id)
    candidate_id: str
    consent_type: Literal["behavior_signal"] = "behavior_signal"
    granted: bool = True
    granted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    revoked_at: datetime | None = None

    @field_validator("id", "candidate_id")
    @classmethod
    def identifiers_are_not_blank(cls, value: str, info: ValidationInfo) -> str:
        return _not_blank(value, info.field_name)


class InterviewCreate(BaseModel):
    job_id: str
    candidate_id: str
    signal_enabled: bool = False

    @field_validator("job_id", "candidate_id")
    @classmethod
    def identifiers_are_not_blank(cls, value: str, info: ValidationInfo) -> str:
        return _not_blank(value, info.field_name)


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

    @field_validator("id", "job_id", "candidate_id")
    @classmethod
    def identifiers_are_not_blank(cls, value: str, info: ValidationInfo) -> str:
        return _not_blank(value, info.field_name)


class OfflineInterviewInput(BaseModel):
    job_title: str
    jd_text: str
    candidate_name: str
    resume_text: str = ""
    turns: list[QATurn] = Field(min_length=1)

    @field_validator("job_title", "jd_text", "candidate_name")
    @classmethod
    def text_fields_are_not_blank(cls, value: str) -> str:
        return _not_blank(value, "offline input text")


class AIGCDetectRequest(BaseModel):
    turns: list[QATurn] = Field(min_length=1)


class ScoringRequest(BaseModel):
    context: InterviewContext
    aigc_results: list[AIGCResult] = Field(min_length=1)


class ReportBuildRequest(BaseModel):
    context: InterviewContext
    score: InterviewScore
    aigc_results: list[AIGCResult] = Field(min_length=1)


class OfflineInterviewResult(BaseModel):
    job: JobRecord
    candidate: CandidateRecord
    interview: InterviewRecord
    report: "Report"


class OfflineTaskAccepted(BaseModel):
    interview_id: str
    task_id: str
    task_name: str
    status: Literal["queued"]
    message: str = "offline scoring task queued"

    @field_validator("interview_id", "task_id", "task_name")
    @classmethod
    def identifiers_are_not_blank(cls, value: str, info: ValidationInfo) -> str:
        return _not_blank(value, info.field_name)


class Report(BaseModel):
    interview_id: str
    score: InterviewScore
    aigc_results: list[AIGCResult] = Field(min_length=1)
    consistency_flags: list[ConsistencyFlag]
    transcript: list[QATurn] = Field(default_factory=list)
    summary: str
    json_path: str | None = None
    html_path: str | None = None
    pdf_path: str | None = None
    transcript_path: str | None = None
    artifact_uris: dict[str, str] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("interview_id")
    @classmethod
    def interview_id_is_not_blank(cls, value: str) -> str:
        return _not_blank(value, "interview_id")

    @field_validator("summary")
    @classmethod
    def summary_is_not_blank(cls, value: str) -> str:
        return _not_blank(value, "summary")
