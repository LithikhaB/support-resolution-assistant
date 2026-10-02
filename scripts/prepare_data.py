"""Prepare historical responses and an auditable data-quality manifest."""
import json
from collections import Counter
from datetime import datetime, timezone

from app.config.settings import get_settings
from app.ingestion.artifacts import atomic_write, file_sha256, write_json
from app.ingestion.loader import PIPELINE_VERSION, SOURCE, load_tickets
from app.monitoring.logging import configure_logging


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    source = settings.raw_dir / "customer_support_tickets.csv"
    source_hash = file_sha256(source)
    source_manifest = source.with_suffix(".manifest.json")
    provenance = {"revision": None, "provenance_status": "legacy_local_snapshot"}
    if source_manifest.exists():
        provenance = json.loads(source_manifest.read_text(encoding="utf-8"))
        if provenance.get("sha256") != source_hash:
            raise ValueError("Raw dataset checksum differs from its source manifest")
    tickets, drops = load_tickets(source)
    if not tickets:
        raise ValueError("No valid records; refusing to replace processed artifacts")
    if file_sha256(source) != source_hash:
        raise ValueError("Raw dataset changed while being processed")
    out = settings.processed_dir / "documents.jsonl"
    atomic_write(out, (doc.model_dump_json() + "\n" for doc in tickets))
    report = {
        "pipeline_version": PIPELINE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": SOURCE, "source_revision": provenance.get("revision"),
        "source_provenance_status": provenance.get("provenance_status", "pinned_download"),
        "source_sha256": source_hash, "output_sha256": file_sha256(out),
        "documents": len(tickets), "dropped": dict(drops),
        "input_rows": len(tickets) + sum(drops.values()),
        "doc_types": dict(Counter(d.doc_type.value for d in tickets)),
        "outcome_status": dict(Counter(d.outcome_status for d in tickets)),
        "ticket_types": dict(Counter(d.ticket_type for d in tickets)),
        "priorities": dict(Counter(d.priority.value if d.priority else "unknown" for d in tickets)),
        "complaint_groups": len({d.metadata["complaint_group_id"] for d in tickets}),
        "literal_newlines_remaining": sum("\\n" in d.body or "\\n" in d.response for d in tickets),
        "notes": ["Replies are not verified resolutions.",
                  "Intent, severity, product and sentiment are unknown in this source.",
                  "Content IDs survive reordering, but edits create new identities.",
                  "Complaint groups support exact-duplicate splits; near-duplicate splitting remains future work."],
    }
    write_json(settings.processed_dir / "manifest.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
