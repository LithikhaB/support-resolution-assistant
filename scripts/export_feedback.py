"""Export de-identified agent reviews for deliberate dataset curation, never automatic training."""

import argparse
import json
from pathlib import Path

from app.config.settings import get_settings
from app.ingestion.artifacts import atomic_write
from app.llm.privacy import redact
from app.workflow.models import CaseRecord
from app.workflow.store import CaseStore


def feedback_rows(store, limit=1000):
    """Retain rejection and outcome evidence; skip pending cases without agent review."""
    with store.connection() as connection:
        rows = connection.execute(
            "SELECT payload FROM cases ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
    for row in rows:
        record = CaseRecord.model_validate_json(row[0])
        for review in record.reviews:
            issue = next((i for i in review.original_resolution.analysis.products), None)
            payload = {
                "original_complaint": record.request.query,
                "category": review.original_resolution.analysis.category,
                "product": issue.product if issue else None,
                "generation": review.generation,
                "action": review.review.action,
                "outcome": review.review.outcome,
                "notes": review.review.notes,
                "outcome_notes": review.review.outcome_notes,
                "reviewed_text": review.reviewed_text,
                "source_ids": [s.doc_id for s in review.original_resolution.sources],
                "curation_status": "needs_human_validation",
                "eligible_for_automatic_training": False,
            }
            masked, _ = redact(payload)
            yield masked


def main():
    """Write a new local feedback artifact without contacting external services."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output file")
    rows = list(feedback_rows(CaseStore(get_settings().case_store_path)))
    atomic_write(args.output, (json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print(
        f"Exported {len(rows)} review events for human curation. Common identifiers masked; review free text for remaining private details."
    )


if __name__ == "__main__":
    main()
