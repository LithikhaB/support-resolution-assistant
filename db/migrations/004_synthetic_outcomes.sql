ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_outcome_status_check;
ALTER TABLE documents ADD CONSTRAINT documents_outcome_status_check
    CHECK (outcome_status IN ('unknown','verified_resolved','simulated_resolved')) NOT VALID;
ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_evidence_check;
ALTER TABLE documents ADD CONSTRAINT documents_evidence_check CHECK (
    (doc_type='historical_response' AND response IS NOT NULL AND length(trim(response)) > 0
        AND resolution IS NULL AND outcome_status='unknown') OR
    (doc_type='resolved_ticket' AND resolution IS NOT NULL AND length(trim(resolution)) > 0
        AND ((outcome_status='verified_resolved' AND coalesce(metadata->>'is_synthetic','false') <> 'true')
          OR (outcome_status='simulated_resolved' AND coalesce(metadata->'is_synthetic' = 'true'::jsonb,false)
              AND coalesce(length(trim(metadata->>'scenario_family')),0) > 0))
        AND coalesce(length(trim(metadata->>'outcome_evidence')),0) > 0) OR
    (doc_type='knowledge_base' AND outcome_status='unknown')
) NOT VALID;
