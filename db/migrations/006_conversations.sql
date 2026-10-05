CREATE TABLE IF NOT EXISTS support_conversations (
    conversation_id uuid PRIMARY KEY,
    owner_hash text NOT NULL,
    revision integer NOT NULL DEFAULT 1,
    request jsonb NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now(),
    expires_at timestamptz NOT NULL
);
CREATE INDEX IF NOT EXISTS support_conversations_owner ON support_conversations(owner_hash, updated_at DESC);
CREATE TABLE IF NOT EXISTS support_conversation_events (
    conversation_id uuid NOT NULL REFERENCES support_conversations ON DELETE CASCADE,
    revision integer NOT NULL,
    response jsonb NOT NULL,
    source_revision jsonb NOT NULL,
    PRIMARY KEY(conversation_id, revision)
);
CREATE TABLE IF NOT EXISTS support_reviewed_resolutions (
    history_id bigserial PRIMARY KEY,
    conversation_id uuid NOT NULL REFERENCES support_conversations ON DELETE CASCADE,
    owner_hash text NOT NULL,
    issue_id integer NOT NULL CHECK(issue_id BETWEEN 1 AND 4),
    source_id text NOT NULL,
    source_revision jsonb NOT NULL,
    category text,
    complaint text NOT NULL,
    resolution text NOT NULL,
    outcome_note text NOT NULL,
    outcome_status text NOT NULL CHECK(outcome_status = 'simulated_resolved'),
    embedding vector(384) NOT NULL,
    reviewed_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(conversation_id, issue_id, source_id)
);
CREATE INDEX IF NOT EXISTS support_reviewed_owner ON support_reviewed_resolutions(owner_hash);
