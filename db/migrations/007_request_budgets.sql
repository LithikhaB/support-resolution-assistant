-- Shared token budgets across API workers/replicas; keys contain hashes, never complaints.
CREATE TABLE IF NOT EXISTS support_request_budgets (
    bucket_key text PRIMARY KEY,
    tokens double precision NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    blocked_until timestamptz NOT NULL DEFAULT '-infinity'
);
CREATE INDEX IF NOT EXISTS support_request_budgets_expiry
    ON support_request_budgets(updated_at);
