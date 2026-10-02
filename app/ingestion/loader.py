"""Normalize source replies without claiming they are verified resolutions."""
import csv
import hashlib
import json
import logging
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from app.ingestion.cleaner import clean_text, normalize_label, normalize_severity
from app.ingestion.schema import MIN_BODY_CHARS, DocType, SupportDocument

SOURCE = "Tobi-Bueck/customer-support-tickets"
PIPELINE_VERSION = "2.0.0"
logger = logging.getLogger(__name__)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def load_tickets(path: Path) -> tuple[list[SupportDocument], Counter]:
    docs: dict[str, SupportDocument] = {}
    dropped: Counter = Counter()
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"body", "answer", "language", "type", "priority", "queue"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")
        for row_number, row in enumerate(reader, start=2):
            language = (row.get("language") or "").strip().lower()
            if language != "en":
                dropped["non_english"] += 1
                continue
            body, answer = clean_text(row.get("body")), clean_text(row.get("answer"))
            if len(body) < MIN_BODY_CHARS:
                dropped["body_too_short"] += 1
                continue
            if not answer:
                dropped["missing_response"] += 1
                continue
            title = clean_text(row.get("subject")) or body[:60]
            ticket_type = normalize_label(row.get("type"))
            queue = normalize_label(row.get("queue"))
            raw_priority = normalize_label(row.get("priority"))
            priority = normalize_severity(row.get("priority"))
            tags = sorted({clean_text(row.get(f"tag_{i}")) for i in range(1, 9)} - {""})
            # No upstream immutable ticket ID exists. Content-derived IDs survive
            # row reordering; changed content becomes a new record, not an update.
            content = {"source": SOURCE, "title": title, "body": body, "response": answer,
                       "ticket_type": ticket_type, "queue": queue, "priority": raw_priority,
                       "tags": tags, "language": language}
            content_hash = digest(content)
            doc_id = f"hf_{content_hash}"
            provenance = {"source_row": row_number, "source_version": row.get("version") or None}
            if doc_id in docs:
                docs[doc_id].metadata["provenance"].append(provenance)
                dropped["duplicate_record"] += 1
                continue
            try:
                docs[doc_id] = SupportDocument(
                    doc_id=doc_id, doc_type=DocType.HISTORICAL_RESPONSE,
                    title=title, body=body, response=answer,
                    ticket_type=ticket_type, priority=priority,
                    metadata={"source": SOURCE, "language": language, "queue": queue,
                              "tags": tags, "source_priority": raw_priority,
                              "provenance": [provenance], "content_sha256": content_hash,
                              "complaint_group_id": digest(body.casefold()),
                              "pipeline_version": PIPELINE_VERSION,
                              "outcome_basis": "Source contains a reply, not a verified outcome"},
                )
            except ValidationError:
                dropped["invalid_schema"] += 1
    logger.info("Historical replies loaded=%d dropped=%s", len(docs), dict(dropped))
    return sorted(docs.values(), key=lambda d: d.doc_id), dropped
