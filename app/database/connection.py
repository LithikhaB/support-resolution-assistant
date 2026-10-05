import atexit
from collections.abc import Iterator
from contextlib import contextmanager
from threading import Lock

import psycopg
from psycopg_pool import ConnectionPool

from app.config.settings import get_settings

_pools = {}
_lock = Lock()


def get_pool(settings):
    """Reuse a bounded pool per database configuration, initialized without network I/O."""
    key = (
        settings.postgres_host,
        settings.postgres_port,
        settings.postgres_db,
        settings.postgres_user,
        settings.postgres_password,
        settings.postgres_pool_size,
    )
    with _lock:
        if key not in _pools:
            # Operationally the API has one configuration. Bound experiment/test configs too.
            if len(_pools) >= 4:
                _pools.pop(next(iter(_pools))).close()
            _pools[key] = ConnectionPool(
                kwargs=dict(
                    host=settings.postgres_host,
                    port=settings.postgres_port,
                    dbname=settings.postgres_db,
                    user=settings.postgres_user,
                    password=settings.postgres_password,
                    connect_timeout=settings.postgres_connect_timeout,
                ),
                min_size=0,
                max_size=settings.postgres_pool_size,
                timeout=settings.postgres_pool_timeout_seconds,
                max_waiting=16,
                check=ConnectionPool.check_connection,
                name="support-db",
                open=True,
            )
        return _pools[key]


def close_pools():
    with _lock:
        for pool in _pools.values():
            pool.close()
        _pools.clear()


def pool_stats():
    """Expose aggregate resource measurements without database identities or credentials."""
    with _lock:
        values = [pool.get_stats() for pool in _pools.values()]
    return {
        key: sum(item.get(key, 0) for item in values)
        for key in (
            "pool_size",
            "pool_available",
            "requests_waiting",
            "requests_num",
            "requests_errors",
            "connections_num",
        )
    }


atexit.register(close_pools)


@contextmanager
def get_connection(
    *, statement_timeout_ms: int | None = None, pooled: bool = True
) -> Iterator[psycopg.Connection]:
    """Open a Postgres connection; commits on success, rolls back on error."""
    settings = get_settings()
    timeout = (
        settings.postgres_statement_timeout_ms
        if statement_timeout_ms is None
        else statement_timeout_ms
    )
    if type(timeout) is not int or timeout < 1:
        raise ValueError("statement_timeout_ms must be a positive integer")
    if not pooled:
        # The indexer owns session locks and transaction boundaries for durable checkpoints.
        with psycopg.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            dbname=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password,
            connect_timeout=settings.postgres_connect_timeout,
            options=f"-c statement_timeout={timeout}",
        ) as conn:
            yield conn
        return
    with get_pool(settings).connection() as conn:
        with conn.transaction():
            conn.execute("SELECT set_config('statement_timeout',%s,true)", (str(timeout),))
            yield conn
