CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE probe_patterns
ADD COLUMN IF NOT EXISTS embedding_vector vector(64);

CREATE INDEX IF NOT EXISTS idx_probe_patterns_embedding_vector
ON probe_patterns
USING ivfflat (embedding_vector vector_cosine_ops)
WITH (lists = 16);
