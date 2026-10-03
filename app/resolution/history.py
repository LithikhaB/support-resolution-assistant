"""Expose resolved case evidence without promoting historical outcomes to current diagnoses."""

from pydantic import BaseModel


class HistoricalCase(BaseModel):
    """Preserve the retrieved parent resolution and its outcome provenance verbatim."""

    citation_id: str
    doc_id: str
    chunk_id: int
    title: str
    resolution: str
    outcome_status: str
    is_synthetic: bool


def select_history(results, sources, limit=3):
    """Use resolved cases linked to an applicable KB, excluding unverified outcomes."""
    allowed = {source.doc_id for source in sources}
    selected, seen = [], set()
    for row in results:
        synthetic = row.metadata.get("is_synthetic") is True
        refs = row.metadata.get("kb_refs", [])
        if not isinstance(refs, list) or any(not isinstance(ref, str) for ref in refs):
            continue
        if (
            row.doc_type != "resolved_ticket"
            or not row.resolution
            or row.doc_id in seen
            or row.outcome_status not in {"verified_resolved", "simulated_resolved"}
            or (row.outcome_status == "simulated_resolved" and not synthetic)
            or (row.outcome_status == "verified_resolved" and synthetic)
            or not allowed.intersection(refs)
            or not row.metadata.get("outcome_evidence")
        ):
            continue
        seen.add(row.doc_id)
        selected.append(
            HistoricalCase(
                citation_id=f"T{len(selected) + 1}",
                doc_id=row.doc_id,
                chunk_id=row.chunk_id,
                title=row.title,
                resolution=row.resolution,
                outcome_status=row.outcome_status,
                is_synthetic=synthetic,
            )
        )
        if len(selected) == limit:
            break
    return selected
