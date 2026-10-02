from collections.abc import Iterator
from contextlib import contextmanager

import psycopg

from app.config.settings import get_settings


@contextmanager
def get_connection(*, statement_timeout_ms: int | None = None) -> Iterator[psycopg.Connection]:
    """Open a Postgres connection; commits on success, rolls back on error."""
    settings = get_settings()
    timeout = settings.postgres_statement_timeout_ms if statement_timeout_ms is None else statement_timeout_ms
    if not isinstance(timeout, int) or timeout < 1:
        raise ValueError("statement_timeout_ms must be a positive integer")
    with psycopg.connect(
        host=settings.postgres_host, port=settings.postgres_port,
        dbname=settings.postgres_db, user=settings.postgres_user,
        password=settings.postgres_password,
        connect_timeout=settings.postgres_connect_timeout,
        options=f"-c statement_timeout={timeout}",
    ) as conn:
        yield conn