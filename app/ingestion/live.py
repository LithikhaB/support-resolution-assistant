"""Publish a staged corpus and one transactional index under the existing writer lock."""

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import psycopg

from app.config.settings import get_settings
from app.database.connection import get_connection
from app.ingestion.artifacts import atomic_write, file_sha256
from app.ingestion.chunking import chunk_corpus
from app.ingestion.indexing import index_corpus, index_session
from app.ingestion.updates import stage_update
from app.understanding.service import get_understanding_service

ARTIFACTS = (
    "documents.jsonl",
    "manifest.json",
    "chunks.jsonl",
    "chunks.manifest.json",
    "indexing.manifest.json",
)
PREDICTION_NOTE = (
    "Retrieval is updated immediately. New category prediction requires labeled train/dev "
    "examples and a category-product mapping, then python -m scripts train && python -m scripts calibrate."
)


class IngestValidationError(ValueError):
    pass


class IngestUnavailable(RuntimeError):
    pass


@contextmanager
def writer(settings):
    with get_connection(
        statement_timeout_ms=settings.indexing_statement_timeout_ms, pooled=False
    ) as conn:
        conn.autocommit = True
        if not conn.execute("SELECT pg_try_advisory_lock(8041, 2)").fetchone()[0]:
            raise IngestUnavailable("index busy")
        try:
            yield conn
        finally:
            # Closing this unpooled connection also releases the session lock.
            # A lost connection after commit must not turn a published update into a failure.
            try:
                conn.execute("SELECT pg_advisory_unlock(8041, 2)")
            except psycopg.Error:
                pass


def index_revision(conn):
    row = conn.execute(
        "SELECT status,source_hash,chunks_hash,config_hash,completed_at FROM retrieval_index_state WHERE singleton"
    ).fetchone()
    if not row or row[0] != "ready":
        raise IngestUnavailable("index unavailable")
    return tuple(row[1:4]) + (str(row[4]),)


def publish_artifacts(staged, active):
    for name in ARTIFACTS:
        atomic_write(active / name, [(staged / name).read_text(encoding="utf-8")])


def restore_artifacts(backups, active):
    for name, content in backups.items():
        if content is None:
            (active / name).unlink(missing_ok=True)
        else:
            atomic_write(active / name, [content.decode("utf-8")])


def ingest_documents(documents, settings=None):
    """Retrieval updates immediately; new-class prediction needs training/calibration.

    CLI indexing retains resumable commits; here its nested writes are savepoints in
    one outer transaction. No active files are written until indexing succeeds.
    Files are restored on any publication/transaction failure before releasing the lock.
    """
    settings = settings or get_settings()
    started = perf_counter()
    classes = set(get_understanding_service().classifier.artifact.classes)
    new_categories = sorted(
        {doc.intent for doc in documents if doc.intent and doc.intent not in classes}
    )
    active = settings.processed_dir
    with writer(settings) as conn:
        before = index_revision(conn)
        if (
            file_sha256(active / "documents.jsonl"),
            file_sha256(active / "chunks.jsonl"),
        ) != before[:2]:
            raise IngestUnavailable("active corpus differs from index")
        with TemporaryDirectory(prefix="ingest-", dir=settings.corpus_dir.parent) as temporary:
            root = Path(temporary)
            incoming = root / "incoming.jsonl"
            atomic_write(incoming, (doc.model_dump_json() + "\n" for doc in documents))
            staged = root / "processed"
            try:
                changes = stage_update(active / "documents.jsonl", incoming, staged)
            except ValueError as exc:
                raise IngestValidationError("invalid corpus update") from exc
            candidate = settings.model_copy(update={"corpus_dir": root})
            chunk_corpus(candidate)
            backups = {
                name: (active / name).read_bytes() if (active / name).exists() else None
                for name in ARTIFACTS
            }
            token = index_session.set(conn)
            publication_started = False
            try:
                with conn.transaction():
                    report = index_corpus(candidate)
                    conn.execute(
                        "UPDATE retrieval_index_state SET completed_at=clock_timestamp() WHERE singleton"
                    )
                    revision = index_revision(conn)
                    if revision == before:
                        raise IngestUnavailable("revision was not updated")
                    publication_started = True
                    publish_artifacts(staged, active)
            except BaseException:
                if publication_started:
                    restore_artifacts(backups, active)
                raise
            finally:
                index_session.reset(token)
    return {
        "added": changes["added"],
        "updated": changes["updated"],
        "embeddings_generated": report["embeddings_generated"],
        "new_categories": new_categories,
        "classifier_retrain_required": bool(new_categories),
        "category_prediction_note": PREDICTION_NOTE,
        "revision": list(revision),
        "elapsed_ms": (perf_counter() - started) * 1000,
    }
