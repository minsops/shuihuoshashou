CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL CHECK (btrim(title) <> ''),
    jd_text TEXT CHECK (jd_text IS NULL OR btrim(jd_text) <> ''),
    competency_model JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS candidates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT CHECK (name IS NULL OR btrim(name) <> ''),
    resume_text TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS interviews (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES jobs(id),
    candidate_id UUID NOT NULL REFERENCES candidates(id),
    status TEXT NOT NULL CHECK (status IN ('CREATED', 'IN_PROGRESS', 'FINISHED', 'SCORING', 'REPORTED')),
    context JSONB NOT NULL,
    signal_enabled BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,
    CHECK (ended_at IS NULL OR started_at IS NULL OR ended_at >= started_at)
);

CREATE TABLE IF NOT EXISTS qa_turns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    interview_id UUID NOT NULL REFERENCES interviews(id) ON DELETE CASCADE,
    turn_index INT NOT NULL CHECK (turn_index >= 0),
    question TEXT NOT NULL CHECK (btrim(question) <> ''),
    question_source TEXT NOT NULL CHECK (question_source IN ('interviewer', 'ai_probe')),
    answer TEXT NOT NULL CHECK (btrim(answer) <> ''),
    answer_start_ms INT NOT NULL CHECK (answer_start_ms >= 0),
    answer_end_ms INT NOT NULL CHECK (answer_end_ms >= answer_start_ms),
    probe_target TEXT CHECK (probe_target IS NULL OR btrim(probe_target) <> ''),
    payload JSONB NOT NULL,
    UNIQUE (interview_id, turn_index)
);

CREATE TABLE IF NOT EXISTS probe_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    competency TEXT NOT NULL CHECK (btrim(competency) <> ''),
    pattern TEXT NOT NULL CHECK (btrim(pattern) <> ''),
    embedding JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS scores (
    interview_id UUID PRIMARY KEY REFERENCES interviews(id) ON DELETE CASCADE,
    dimensions JSONB NOT NULL CHECK (
        jsonb_typeof(dimensions) = 'array' AND jsonb_array_length(dimensions) > 0
    ),
    total_score REAL NOT NULL CHECK (total_score >= 0 AND total_score <= 100),
    risk_notes JSONB NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(risk_notes) = 'array'),
    recommendation TEXT NOT NULL CHECK (recommendation IN ('strong_yes', 'yes', 'hold', 'no')),
    payload JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS aigc_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    interview_id UUID NOT NULL REFERENCES interviews(id) ON DELETE CASCADE,
    turn_id UUID NOT NULL REFERENCES qa_turns(id) ON DELETE CASCADE,
    ai_generated_prob REAL NOT NULL CHECK (ai_generated_prob >= 0 AND ai_generated_prob <= 1),
    template_similarity REAL NOT NULL CHECK (template_similarity >= 0 AND template_similarity <= 1),
    matched_template TEXT CHECK (matched_template IS NULL OR btrim(matched_template) <> ''),
    flagged BOOLEAN NOT NULL DEFAULT false,
    payload JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS reports (
    interview_id UUID PRIMARY KEY REFERENCES interviews(id) ON DELETE CASCADE,
    payload JSONB NOT NULL,
    html TEXT NOT NULL CHECK (btrim(html) <> ''),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS consents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    candidate_id UUID NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
    consent_type TEXT NOT NULL CHECK (consent_type IN ('behavior_signal')),
    granted BOOLEAN NOT NULL,
    granted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    revoked_at TIMESTAMPTZ,
    CHECK (revoked_at IS NULL OR revoked_at >= granted_at)
);

CREATE INDEX IF NOT EXISTS idx_interviews_job_id ON interviews(job_id);
CREATE INDEX IF NOT EXISTS idx_interviews_candidate_id ON interviews(candidate_id);
CREATE INDEX IF NOT EXISTS idx_qa_turns_interview_id ON qa_turns(interview_id);
CREATE INDEX IF NOT EXISTS idx_probe_patterns_job_id ON probe_patterns(job_id);
CREATE INDEX IF NOT EXISTS idx_aigc_results_interview_id ON aigc_results(interview_id);
CREATE INDEX IF NOT EXISTS idx_consents_candidate_type ON consents(candidate_id, consent_type);
