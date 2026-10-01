CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    doc_type    TEXT NOT NULL CHECK (doc_type IN ('resolved_ticket', 'knowledge_base')),
    title       TEXT NOT NULL,
    body        TEXT NOT NULL,
    resolution  TEXT,
    intent      TEXT,
    product     TEXT,
    severity    TEXT,
    sentiment   TEXT,
    created_at  DATE,
    metadata    JSONB NOT NULL DEFAULT '{}',
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 384 = output size of all-MiniLM-L6-v2 
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    BIGSERIAL PRIMARY KEY,
    doc_id      TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_index INT  NOT NULL,
    content     TEXT NOT NULL,
    embedding   vector(384),
    UNIQUE (doc_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_documents_intent  ON documents(intent);
CREATE INDEX IF NOT EXISTS idx_documents_product ON documents(product);
CREATE INDEX IF NOT EXISTS idx_documents_type    ON documents(doc_type);
