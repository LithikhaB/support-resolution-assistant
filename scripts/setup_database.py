"""Create the configured database if missing and initialize its schema safely."""

from pathlib import Path

import psycopg
from psycopg import sql

from app.config.settings import get_settings
from app.database.connection import get_connection


def main() -> None:
    """Create a separate database without dropping or replacing an existing corpus."""
    settings = get_settings()
    with psycopg.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        dbname="postgres",
        user=settings.postgres_user,
        password=settings.postgres_password,
        connect_timeout=settings.postgres_connect_timeout,
        autocommit=True,
    ) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname=%s", (settings.postgres_db,)
        ).fetchone()
        if not exists:
            try:
                conn.execute(
                    sql.SQL("CREATE DATABASE {}").format(sql.Identifier(settings.postgres_db))
                )
            except psycopg.errors.DuplicateDatabase:
                pass
    root = Path(__file__).resolve().parents[1]
    with get_connection(statement_timeout_ms=60000) as conn:
        conn.execute((root / "db/init.sql").read_text(encoding="utf-8"))
        for name in (
            "003_retrieval_indexing.sql",
            "004_synthetic_outcomes.sql",
            "005_shared_lexical.sql",
        ):
            conn.execute((root / "db/migrations" / name).read_text(encoding="utf-8"))
    print(f"Database {settings.postgres_db} initialized; existing rows preserved.")


if __name__ == "__main__":
    main()
