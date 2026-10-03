"""Small transactional SQLite store independent of the retrieval corpus and training data."""

import sqlite3
from contextlib import contextmanager

from app.workflow.models import CaseRecord


class RevisionConflict(Exception):
    """The client reviewed an older case revision."""


class CaseStore:
    """Save case snapshots atomically, with optimistic concurrency across processes."""

    def __init__(self, path):
        self.path = path

    @contextmanager
    def connection(self):
        """Initialize storage lazily and always close the per-operation connection."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=5)
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cases (id TEXT PRIMARY KEY, revision INTEGER NOT NULL, updated_at TEXT NOT NULL, payload TEXT NOT NULL)"
            )
            with conn:
                yield conn
        finally:
            conn.close()

    def create(self, record):
        """Insert a new server-generated case without replacing existing records."""
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO cases VALUES (?, ?, ?, ?)",
                (
                    str(record.id),
                    record.revision,
                    record.updated_at.isoformat(),
                    record.model_dump_json(),
                ),
            )
        return record

    def get(self, case_id):
        """Read a validated snapshot or signal that the case does not exist."""
        with self.connection() as conn:
            row = conn.execute("SELECT payload FROM cases WHERE id = ?", (str(case_id),)).fetchone()
        if row is None:
            raise KeyError(case_id)
        return CaseRecord.model_validate_json(row[0])

    def save(self, record, expected_revision):
        """Commit only if the version used to prepare this change is still current."""
        with self.connection() as conn:
            changed = conn.execute(
                "UPDATE cases SET revision = ?, updated_at = ?, payload = ? WHERE id = ? AND revision = ?",
                (
                    record.revision,
                    record.updated_at.isoformat(),
                    record.model_dump_json(),
                    str(record.id),
                    expected_revision,
                ),
            ).rowcount
            if changed != 1:
                raise RevisionConflict()
        return record

    def recent(self, limit=30):
        """Return bounded summaries without loading models or exposing full transcripts."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT payload FROM cases ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        records = [CaseRecord.model_validate_json(row[0]) for row in rows]
        return [
            {
                "id": str(c.id),
                "revision": c.revision,
                "updated_at": c.updated_at,
                "title": c.request.query[:100],
                "issues": len(c.response.issues),
            }
            for c in records
        ]
