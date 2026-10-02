CREATE TABLE IF NOT EXISTS document_index_state (
    doc_id TEXT PRIMARY KEY REFERENCES documents(doc_id) ON DELETE CASCADE,
    fingerprint TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    chunk_count INTEGER NOT NULL CHECK (chunk_count > 0),
    indexed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS retrieval_index_state (
    singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
    status TEXT NOT NULL CHECK (status IN ('indexing', 'ready')),
    config_hash TEXT NOT NULL,
    configuration JSONB NOT NULL,
    source_hash TEXT NOT NULL,
    chunks_hash TEXT NOT NULL,
    completed_at TIMESTAMPTZ
);
