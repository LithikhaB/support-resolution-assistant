from collections.abc import Iterator
from contextlib import contextmanager

import psycopg

from app.config.settings import get_settings


@contextmanager
def get_connection() -> Iterator[psycopg.Connection]:
    """Open a Postgres connection; commits on success, rolls back on error."""
    with psycopg.connect(get_settings().database_url) as conn:
        yield conn