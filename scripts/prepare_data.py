"""Clean and normalize the Hugging Face support dataset into
data/processed/documents.jsonl.
"""

from app.config.settings import get_settings
from app.ingestion.loader import load_tickets
from app.monitoring.logging import configure_logging


def main() -> None:
    settings = get_settings()

    configure_logging(settings.log_level)

    settings.processed_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Load and normalize the Hugging Face dataset
    # ---------------------------------------------------------

    tickets, ticket_drops = load_tickets(
        settings.raw_dir / "customer_support_tickets.csv"
    )

    # ---------------------------------------------------------
    # Write normalized documents
    # ---------------------------------------------------------

    out_path = settings.processed_dir / "documents.jsonl"

    with out_path.open("w", encoding="utf-8") as f:
        for doc in tickets:
            f.write(doc.model_dump_json() + "\n")

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print(f"\nwrote {len(tickets)} documents -> {out_path}")

    print(f"  tickets: {len(tickets)}")
    print(f"  dropped: {dict(ticket_drops)}")


if __name__ == "__main__":
    main()