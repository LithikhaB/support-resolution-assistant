from collections.abc import Iterator
from contextlib import contextmanager

import psycopg

from app.config.settings import get_settings


@contextmanager
def get_connection() -> Iterator[psycopg.Connection]:
    """Open a Postgres connection; commits on success, rolls back on error."""
    settings = get_settings()
    with psycopg.connect(
        host=settings.postgres_host, port=settings.postgres_port,
        dbname=settings.postgres_db, user=settings.postgres_user,
        password=settings.postgres_password,
        connect_timeout=settings.postgres_connect_timeout,
        options="-c statement_timeout=3000",
    ) as conn:
        yield conn