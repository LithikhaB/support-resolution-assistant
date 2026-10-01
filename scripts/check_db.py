"""Clean + normalize raw data into data/processed/documents.jsonl."""
from app.config.settings import get_settings
from app.ingestion.loader import load_kb_articles, load_tickets
from app.monitoring.logging import configure_logging


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)

    tickets, ticket_drops = load_tickets(settings.raw_dir / "tickets.csv")
    articles, kb_drops = load_kb_articles(settings.raw_dir / "kb_articles.json")

    out_path = settings.processed_dir / "documents.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for doc in [*articles, *tickets]:
            f.write(doc.model_dump_json() + "\n")

    print(f"\nwrote {len(tickets) + len(articles)} documents -> {out_path}")
    print(f"  tickets: {len(tickets)}  dropped: {dict(ticket_drops)}")
    print(f"  kb:      {len(articles)}  dropped: {dict(kb_drops)}")


if __name__ == "__main__":
    main()