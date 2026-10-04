"""Opt in: RUN_DB_TESTS=1. Every test schema and write is rolled back."""

import os

import psycopg
import pytest

from app.ingestion.schema import SupportDocument
from app.retrieval.chunking import Chunk
from app.retrieval.corpus import CorpusDocument

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_DB_TESTS") != "1", reason="Set RUN_DB_TESTS=1 for PostgreSQL tests"
)


def item(body="The broadband connection keeps dropping.", split=False):
    doc = SupportDocument(
        doc_id="test",
        doc_type="historical_response",
        title="Outage",
        body=body,
        response="Please confirm the model.",
    )
    text = doc.title + "\n\n" + doc.body
    chunks = [Chunk("test", 0, text, 10, 0, len(text))]
    if split:
        chunks.append(Chunk("test", 1, "extra chunk", 4, 0, 11))
    return CorpusDocument(doc, chunks)


def vectors(count):
    return [[1.0] + [0.0] * 383 for _ in range(count)]


def begin(repo):
    repo.begin(
        {"model": "test"}, "config", {"source_sha256": "docs", "output_sha256": "chunks"}, {"test"}
    )


def test_upsert_skip_replace_and_stable_chunk_id(repository):
    repo = repository
    begin(repo)
    record = item(split=True)
    repo.write_batch([record], vectors(2), "config")
    first_id = repo.conn.execute("SELECT chunk_id FROM chunks WHERE chunk_index=0").fetchone()[0]
    assert repo.unchanged([record], "config") == {"test"}
    changed = item(body="A changed broadband complaint with different symptoms.")
    repo.write_batch([changed], vectors(1), "config")
    assert repo.conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 1
    assert repo.conn.execute("SELECT chunk_id FROM chunks").fetchone()[0] == first_id
    assert repo.conn.execute(
        "SELECT response,outcome_status,resolution FROM documents"
    ).fetchone() == ("Please confirm the model.", "unknown", None)
    repo.conn.execute("UPDATE chunks SET embedding=NULL")
    assert repo.unchanged([changed], "config") == set()


def test_failed_chunk_write_rolls_back_document_and_checkpoint(repository):
    repo = repository
    begin(repo)
    original = item()
    repo.write_batch([original], vectors(1), "config")
    repo.conn.execute("ALTER TABLE chunks ADD CHECK (content NOT LIKE '%reject%')")
    with pytest.raises(psycopg.errors.CheckViolation):
        repo.write_batch(
            [item(body="reject this deliberately invalid update")], vectors(1), "config"
        )
    assert repo.conn.execute("SELECT body FROM documents").fetchone()[0] == original.document.body
    assert repo.unchanged([original], "config") == {"test"}


def test_hnsw_and_migration_are_rerunnable(repository):
    repo = repository
    repo.migrate()
    begin(repo)
    repo.write_batch([item()], vectors(1), "config")
    repo.finish(1, 1)
    repo.finish(1, 1)
    assert repo.conn.execute("SELECT status FROM retrieval_index_state").fetchone()[0] == "ready"
    row = repo.conn.execute(
        "SELECT embedding <=> %s::vector FROM chunks",
        ("[" + ",".join(map(str, vectors(1)[0])) + "]",),
    ).fetchone()
    assert row[0] == pytest.approx(0)


def test_incompatible_profiles_and_removed_documents_rejected(repository):
    repo = repository
    begin(repo)
    repo.write_batch([item()], vectors(1), "config")
    with pytest.raises(ValueError, match="different configuration"):
        repo.begin({}, "other", {"source_sha256": "a", "output_sha256": "b"}, {"test"})
    with pytest.raises(ValueError, match="omits"):
        repo.begin({}, "config", {"source_sha256": "a", "output_sha256": "b"}, set())


def test_finish_refuses_partial_corpus(repository):
    repo = repository
    begin(repo)
    repo.write_batch([item()], vectors(1), "config")
    with pytest.raises(ValueError, match="counts"):
        repo.finish(2, 2)
    assert repo.conn.execute("SELECT status FROM retrieval_index_state").fetchone()[0] == "indexing"


def test_configuration_rebuild_preserves_evidence_and_resumes(repository):
    repo = repository
    begin(repo)
    record = item()
    repo.write_batch([record], vectors(1), "config")
    repo.finish(1, 1)
    chunk_id = repo.conn.execute("SELECT chunk_id FROM chunks").fetchone()[0]
    manifest = {"source_sha256": "docs", "output_sha256": "chunks"}
    with pytest.raises(ValueError, match="omits"):
        repo.begin({}, "container", manifest, set(), rebuild=True)
    repo.begin({"model": "test"}, "container", manifest, {"test"}, rebuild=True)
    assert repo.conn.execute("SELECT status FROM retrieval_index_state").fetchone()[0] == "indexing"
    assert repo.unchanged([record], "container") == set()
    repo.write_batch([record], vectors(1), "container")
    # A restarted bootstrap skips batches already regenerated with this configuration.
    repo.begin({"model": "test"}, "container", manifest, {"test"}, rebuild=True)
    assert repo.unchanged([record], "container") == {"test"}
    repo.finish(1, 1)
    assert repo.conn.execute("SELECT chunk_id FROM chunks").fetchone()[0] == chunk_id
    assert (
        repo.conn.execute("SELECT response FROM documents").fetchone()[0]
        == record.document.response
    )
    assert repo.conn.execute("SELECT status FROM retrieval_index_state").fetchone()[0] == "ready"
