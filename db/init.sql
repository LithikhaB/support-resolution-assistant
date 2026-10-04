CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    doc_type    TEXT NOT NULL CHECK (doc_type IN ('historical_response', 'resolved_ticket', 'knowledge_base')),
    title       TEXT NOT NULL,
    body        TEXT NOT NULL,
    resolution  TEXT,
    response    TEXT,
    outcome_status TEXT NOT NULL DEFAULT 'unknown' CHECK (outcome_status IN ('unknown', 'verified_resolved', 'simulated_resolved')),
    ticket_type TEXT,
    priority    TEXT,
    intent      TEXT,
    product     TEXT,
    severity    TEXT,
    sentiment   TEXT,
    created_at  DATE,
    metadata    JSONB NOT NULL DEFAULT '{}',
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT documents_evidence_check CHECK (
        (doc_type = 'historical_response' AND length(trim(response)) > 0 AND response IS NOT NULL
          AND resolution IS NULL AND outcome_status = 'unknown') OR
        (doc_type = 'resolved_ticket' AND length(trim(resolution)) > 0 AND resolution IS NOT NULL
          AND ((outcome_status = 'verified_resolved' AND coalesce(metadata->>'is_synthetic','false') <> 'true')
            OR (outcome_status = 'simulated_resolved' AND coalesce(metadata->'is_synthetic' = 'true'::jsonb,false)
                AND coalesce(length(trim(metadata->>'scenario_family')),0) > 0))
          AND coalesce(length(trim(metadata->>'outcome_evidence')), 0) > 0) OR
        (doc_type = 'knowledge_base' AND outcome_status = 'unknown')
    )
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    BIGSERIAL PRIMARY KEY,
    doc_id      TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_index INT  NOT NULL,
    content     TEXT NOT NULL,
    search_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    embedding   vector(384),
    UNIQUE (doc_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_documents_intent  ON documents(intent);
CREATE INDEX IF NOT EXISTS idx_documents_product ON documents(product);
CREATE INDEX IF NOT EXISTS idx_documents_type    ON documents(doc_type);
CREATE INDEX IF NOT EXISTS idx_chunks_search_vector ON chunks USING gin(search_vector);
