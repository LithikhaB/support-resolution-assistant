import csv
import logging
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from app.ingestion.cleaner import (
    clean_text,
    normalize_label,
    normalize_severity,
)
from app.ingestion.schema import MIN_BODY_CHARS, DocType, SupportDocument

logger = logging.getLogger(__name__)


def load_tickets(
    path: Path,
) -> tuple[list[SupportDocument], Counter]:
    """
    Load the Hugging Face customer-support ticket CSV and convert
    each resolved English ticket into a SupportDocument.

    Expected dataset columns:
        subject
        body
        answer
        type
        queue
        priority
        language
        tag_1 ... tag_8
    """

    docs: list[SupportDocument] = []
    dropped: Counter = Counter()
    seen_bodies: set[str] = set()

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row_number, row in enumerate(reader, start=2):

            # ---------------------------------------------------------
            # 1. Language filtering
            # ---------------------------------------------------------

            language = (row.get("language") or "").strip().lower()

            if language != "en":
                dropped["non_english"] += 1
                continue

            # ---------------------------------------------------------
            # 2. Clean ticket fields
            # ---------------------------------------------------------

            subject = clean_text(row.get("subject"))
            body = clean_text(row.get("body"))
            answer = clean_text(row.get("answer"))

            # ---------------------------------------------------------
            # 3. Validate the ticket
            # ---------------------------------------------------------

            if len(body) < MIN_BODY_CHARS:
                dropped["body_too_short"] += 1
                continue

            if not answer:
                dropped["missing_resolution"] += 1
                continue

            # ---------------------------------------------------------
            # 4. Remove duplicate tickets
            # ---------------------------------------------------------

            body_key = body.lower()

            if body_key in seen_bodies:
                dropped["duplicate"] += 1
                continue

            seen_bodies.add(body_key)

            # ---------------------------------------------------------
            # 5. Normalize metadata
            # ---------------------------------------------------------

            ticket_type = normalize_label(row.get("type"))
            queue = normalize_label(row.get("queue"))
            priority = normalize_severity(row.get("priority"))

            # ---------------------------------------------------------
            # 6. Collect available tags
            # ---------------------------------------------------------

            tags: list[str] = []

            for tag_number in range(1, 9):
                tag_value = row.get(f"tag_{tag_number}")

                if tag_value:
                    tag_value = str(tag_value).strip()

                    if tag_value:
                        tags.append(tag_value)

            # ---------------------------------------------------------
            # 7. Build metadata
            # ---------------------------------------------------------

            metadata = {
                "source": "Tobi-Bueck/customer-support-tickets",
                "language": language,
                "queue": queue,
                "ticket_type": ticket_type,
                "tags": tags,
                "source_row": row_number,
            }

            # ---------------------------------------------------------
            # 8. Create unified SupportDocument
            # ---------------------------------------------------------

            try:
                document = SupportDocument(
                    doc_id=f"hf_ticket_{row_number}",
                    doc_type=DocType.RESOLVED_TICKET,
                    title=subject or body[:60],
                    body=body,
                    resolution=answer,

                    # The dataset provides ticket type and queue,
                    # but does not explicitly provide our own
                    # product/sentiment fields.
                    intent=ticket_type,
                    product=None,
                    severity=priority,
                    sentiment=None,

                    metadata=metadata,
                )

                docs.append(document)

            except ValidationError:
                dropped["invalid_schema"] += 1

    logger.info(
        "Hugging Face tickets loaded=%d dropped=%s",
        len(docs),
        dict(dropped),
    )

    return docs, dropped