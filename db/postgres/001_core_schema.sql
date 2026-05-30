CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    jd_text TEXT,
    competency_model JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS candidates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT,
    resume_text TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS interviews (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES jobs(id),
    candidate_id UUID NOT NULL REFERENCES candidates(id),
    status TEXT NOT NULL,
    context JSONB NOT NULL,
    signal_enabled BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS qa_turns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    interview_id UUID NOT NULL REFERENCES interviews(id) ON DELETE CASCADE,
    turn_index INT NOT NULL,
    question TEXT NOT NULL,
    question_source TEXT NOT NULL,
    answer TEXT NOT NULL,
    answer_start_ms INT NOT NULL,
    answer_end_ms INT NOT NULL,
    probe_target TEXT,
    payload JSONB NOT NULL,
    UNIQUE (interview_id, turn_index)
);

CREATE TABLE IF NOT EXISTS probe_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    competency TEXT NOT NULL,
    pattern TEXT NOT NULL,
    embedding JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS scores (
    interview_id UUID PRIMARY KEY REFERENCES interviews(id) ON DELETE CASCADE,
    dimensions JSONB NOT NULL,
    total_score REAL NOT NULL,
    risk_notes JSONB NOT NULL DEFAULT '[]'::jsonb,
    recommendation TEXT NOT NULL,
    payload JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS aigc_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    interview_id UUID NOT NULL REFERENCES interviews(id) ON DELETE CASCADE,
    turn_id UUID NOT NULL REFERENCES qa_turns(id) ON DELETE CASCADE,
    ai_generated_prob REAL NOT NULL,
    template_similarity REAL NOT NULL,
    matched_template TEXT,
    flagged BOOLEAN NOT NULL DEFAULT false,
    payload JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS reports (
    interview_id UUID PRIMARY KEY REFERENCES interviews(id) ON DELETE CASCADE,
    payload JSONB NOT NULL,
    html TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS consents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    candidate_id UUID NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
    consent_type TEXT NOT NULL,
    granted BOOLEAN NOT NULL,
    granted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    revoked_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_interviews_job_id ON interviews(job_id);
CREATE INDEX IF NOT EXISTS idx_interviews_candidate_id ON interviews(candidate_id);
CREATE INDEX IF NOT EXISTS idx_qa_turns_interview_id ON qa_turns(interview_id);
CREATE INDEX IF NOT EXISTS idx_probe_patterns_job_id ON probe_patterns(job_id);
CREATE INDEX IF NOT EXISTS idx_aigc_results_interview_id ON aigc_results(interview_id);
CREATE INDEX IF NOT EXISTS idx_consents_candidate_type ON consents(candidate_id, consent_type);
