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
    from app.database.connection import get_connection

    with get_connection(statement_timeout_ms=7000, pooled=False) as session:
        session.autocommit = True
        assert session.execute("SHOW statement_timeout").fetchone()[0] == "7s"
        assert session.execute("SELECT pg_try_advisory_lock(9981,1)").fetchone()[0]
        assert session.execute("SELECT pg_advisory_unlock(9981,1)").fetchone()[0]
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

    from contextlib import contextmanager
    from pathlib import Path
    from unittest.mock import Mock
    from uuid import UUID

    from app.resolution.conversation import ConversationRequest, ConversationResponse, IssueResponse
    from app.resolution.drafting import draft_resolution
    from app.resolution.memory import ConversationConflict, ConversationStore
    from app.resolution.validation import finalize_resolution
    from tests.unit.test_agent_triage import v3_source
    from tests.unit.test_resolution import analysis

    repo.conn.execute(Path("db/migrations/006_conversations.sql").read_text(encoding="utf-8"))
    repo.conn.execute(Path("db/migrations/006_conversations.sql").read_text(encoding="utf-8"))

    @contextmanager
    def connection():
        yield repo.conn

    store = ConversationStore(connection=connection)
    query = "My internet is down. Email alice@example.com, password=secret123"
    request = ConversationRequest(query=query)
    observed = analysis(request.query)
    source = v3_source()
    resolution = finalize_resolution(draft_resolution(observed, [source]), [source], observed)
    result = ConversationResponse(
        issues=[
            IssueResponse(
                issue_id=1,
                complaint=request.query,
                analysis_text=request.query,
                resolution=resolution,
            )
        ]
    )
    revision = ("docs", "chunks", "model")
    cid, version = store.save("owner-a", request, result, revision)
    saved = store.load("owner-a", cid)
    assert "alice@example.com" not in str(saved) and "secret123" not in str(saved)
    assert saved["revision"] == version == 1
    assert store.load("owner-b", cid) is None
    assert not store.recent("owner-b")
    request.conversation_id, request.revision = UUID(cid), 1
    assert store.save("owner-a", request, result, revision)[1] == 2
    with pytest.raises(ConversationConflict):
        store.save("owner-a", request, result, revision)
    encoder = Mock()
    encoder.embed_query.return_value = vectors(1)[0]
    assert not store.search_reviewed("owner-a", query, [source.doc_id], revision, encoder)
    with pytest.raises(ConversationConflict):
        store.approve("owner-b", cid, 1, 2, "Synthetic test result reviewed", revision, encoder)
    assert (
        store.approve("owner-a", cid, 1, 2, "Synthetic test result reviewed", revision, encoder)
        == 1
    )
    assert store.search_reviewed("owner-a", query, None, revision, encoder)
    cases = store.search_reviewed("owner-a", query, [source.doc_id], revision, encoder)
    assert len(cases) == 1 and cases[0].outcome_status == "simulated_resolved"
    assert not store.search_reviewed("owner-b", query, [source.doc_id], revision, encoder)
    assert not store.search_reviewed("owner-a", query, [source.doc_id], ("changed",), encoder)
    assert not store.delete("owner-b", cid)
    assert store.delete("owner-a", cid)
    assert not store.search_reviewed("owner-a", query, [source.doc_id], revision, encoder)
