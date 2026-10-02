"""Rollback-only database fixtures shared by integration tests."""

import uuid
from pathlib import Path

import pytest
from psycopg import sql

from app.database.connection import get_connection
from app.database.index_repository import IndexRepository


@pytest.fixture
def repository():
    with get_connection(statement_timeout_ms=60000) as conn:
        with conn.transaction(force_rollback=True):
            namespace = "test_index_" + uuid.uuid4().hex
            conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(namespace)))
            conn.execute(
                sql.SQL("SET LOCAL search_path TO {},public").format(sql.Identifier(namespace))
            )
            conn.execute((Path(__file__).resolve().parents[2] / "db/init.sql").read_text())
            repo = IndexRepository(conn)
            repo.migrate()
            yield repo
