-- Additive migration for an existing Day 1 database; preserves documents/chunks.
BEGIN;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS response TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS outcome_status TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE documents ADD COLUMN IF NOT EXISTS ticket_type TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS priority TEXT;
ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_doc_type_check;
ALTER TABLE documents ADD CONSTRAINT documents_doc_type_check
    CHECK (doc_type IN ('historical_response', 'resolved_ticket', 'knowledge_base'));
-- Legacy HF rows contain unverified replies and proxy labels, not outcomes.
UPDATE documents SET doc_type='historical_response', response=coalesce(response, resolution),
    resolution=NULL, outcome_status='unknown', ticket_type=coalesce(ticket_type, intent),
    priority=coalesce(priority, severity), intent=NULL, severity=NULL
WHERE metadata->>'source' = 'Tobi-Bueck/customer-support-tickets' AND doc_type='resolved_ticket';
-- NOT VALID preserves other legacy rows for review while checking new writes.
ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_evidence_check;
ALTER TABLE documents ADD CONSTRAINT documents_evidence_check CHECK (
    (doc_type='historical_response' AND response IS NOT NULL AND length(trim(response)) > 0
        AND resolution IS NULL AND outcome_status='unknown') OR
    (doc_type='resolved_ticket' AND resolution IS NOT NULL AND length(trim(resolution)) > 0
        AND outcome_status='verified_resolved'
        AND coalesce(length(trim(metadata->>'outcome_evidence')), 0) > 0) OR
    (doc_type='knowledge_base' AND outcome_status='unknown')
) NOT VALID;
COMMIT;
