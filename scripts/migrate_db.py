"""Apply the non-destructive Day 1 evidence migration to an existing database."""
from pathlib import Path

from app.database.connection import get_connection


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "db/migrations"
    with get_connection() as conn:
        # Migration owns its transaction; use the connection context for rollback on error.
        for name in ('002_evidence_semantics.sql', '004_synthetic_outcomes.sql'):
            sql = (directory / name).read_text(encoding="utf-8").replace("BEGIN;", "").replace("COMMIT;", "")
            conn.execute(sql)
    print("Evidence migrations applied; existing records and chunks preserved.")


if __name__ == "__main__":
    main()
