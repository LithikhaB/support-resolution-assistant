"""Apply the non-destructive Day 1 evidence migration to an existing database."""
from pathlib import Path

from app.database.connection import get_connection


def main() -> None:
    migration = Path(__file__).resolve().parents[1] / "db/migrations/002_evidence_semantics.sql"
    with get_connection() as conn:
        # Migration owns its transaction; use the connection context for rollback on error.
        sql = migration.read_text(encoding="utf-8").replace("BEGIN;", "").replace("COMMIT;", "")
        conn.execute(sql)
    print("Day 1 evidence migration applied; existing records and chunks preserved.")


if __name__ == "__main__":
    main()
