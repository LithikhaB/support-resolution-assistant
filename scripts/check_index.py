"""Read-only verification of Phase D rows, embedding norms and HNSW readiness."""

import json

import psycopg

from app.database.connection import get_connection
from app.database.readiness import check_index


def main() -> None:
    """Run the command and report its result to the terminal."""
    try:
        with get_connection(statement_timeout_ms=60000) as conn:
            report = check_index(conn)
    except psycopg.Error:
        print(
            json.dumps(
                {"status": "not_ready", "reason": "Database unavailable or indexing schema missing"}
            )
        )
        raise SystemExit(1)
    print(json.dumps(report, indent=2))
    if report["status"] != "ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
