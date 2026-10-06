"""Live publication is searchable immediately with real PostgreSQL and offline embeddings."""

import json
import os
import shutil
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from tokenizers import Tokenizer, models, pre_tokenizers

from app.api import ingest as api
from app.config.settings import Settings
from app.ingestion import indexing, live
from app.ingestion.chunking import chunk_corpus
from app.main import app
from app.retrieval.cache import published_revision
from app.retrieval.service import RetrievalService, get_retrieval_service
from app.retrieval.vector_search import VectorRetriever

pytestmark = pytest.mark.skipif(os.getenv("RUN_DB_TESTS") != "1", reason="Opt-in database test")


def test_live_dns_ingest_retrieval_revision_and_rollback(repository, tmp_path, monkeypatch):
    settings = Settings(
        _env_file=None, corpus_dir=tmp_path / "corpus", ingest_admin_key="test-admin"
    )
    source = Settings(_env_file=None).processed_dir
    shutil.copytree(source, settings.processed_dir)
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    for module in ("app.ingestion.chunking", "app.ingestion.indexing"):
        monkeypatch.setattr(module + ".load_tokenizer", lambda *args: (tokenizer, path))
    chunk_corpus(settings)
    embedder = Mock()
    embedder.embed_documents.side_effect = lambda texts: [
        [float("DNS" in text), float("DNS" not in text)] + [0.0] * 382 for text in texts
    ]
    embedder.embed_query.return_value = [1.0] + [0.0] * 383
    monkeypatch.setattr(indexing, "get_embedding_service", lambda: embedder)

    @contextmanager
    def connection(*args, **kwargs):
        yield repository.conn

    @contextmanager
    def writer(_):
        repository.conn.execute("SELECT pg_advisory_xact_lock(8041,2)")
        yield repository.conn

    monkeypatch.setattr(api, "get_connection", connection)
    monkeypatch.setattr(live, "writer", writer)
    monkeypatch.setattr("app.database.connection.get_connection", connection)
    monkeypatch.setattr("app.config.settings.get_settings", lambda: settings)
    for module in (api, live):
        monkeypatch.setattr(module, "get_settings", lambda: settings)
        monkeypatch.setattr(
            module,
            "get_understanding_service",
            lambda: SimpleNamespace(
                classifier=SimpleNamespace(artifact=SimpleNamespace(classes=["broadband_outage"]))
            ),
        )
    token = indexing.index_session.set(repository.conn)
    try:
        indexing.index_corpus(settings)
    finally:
        indexing.index_session.reset(token)
    revision = published_revision()
    vector = VectorRetriever(embedder=embedder, connection_factory=connection, settings=settings)
    service = RetrievalService(
        vector=vector, hybrid=Mock(), revision=published_revision, settings=settings
    )
    app.dependency_overrides[get_retrieval_service] = lambda: service
    fixture = json.loads(source.parents[2].joinpath("evolution/dns_category_demo.json").read_text())
    query = {
        "query": fixture["query"],
        "mode": "vector",
        "top_k": 5,
        "filters": {"intent": "broadband_dns"},
    }
    try:
        client = TestClient(app)
        assert client.post("/api/v1/retrieve", json=query).json()["results"] == []
        result = client.post(
            "/api/v1/ingest",
            json={"documents": [fixture["knowledge"]]},
            headers={"X-Admin-Key": "test-admin"},
        )
        assert result.status_code == 200, result.text
        assert result.json()["added"] == 1
        assert result.json()["embeddings_generated"] > 0
        assert published_revision() != revision
        retrieved = client.post("/api/v1/retrieve", json=query)
        assert retrieved.status_code == 200, retrieved.text
        assert retrieved.json()["results"][0]["doc_id"] == fixture["knowledge"]["doc_id"]
        assert client.get("/api/v1/categories").json()["document_counts"]["broadband_dns"] == 1
        revision = published_revision()
        repeated = client.post(
            "/api/v1/ingest",
            json={"documents": [fixture["knowledge"]]},
            headers={"X-Admin-Key": "test-admin"},
        )
        assert repeated.status_code == 200, repeated.text
        assert repeated.json()["updated"] == 1 and repeated.json()["embeddings_generated"] == 0
        assert published_revision() != revision
        revision = published_revision()
        before = {p.name: p.read_bytes() for p in settings.processed_dir.iterdir()}

        def fail(staged, active):
            (active / "documents.jsonl").write_text("partial publication")
            raise OSError("simulated publication failure")

        monkeypatch.setattr(live, "publish_artifacts", fail)
        failed = client.post(
            "/api/v1/ingest",
            json={"documents": [fixture["knowledge"]]},
            headers={"X-Admin-Key": "test-admin"},
        )
        assert failed.status_code == 503
        assert published_revision() == revision
        assert {p.name: p.read_bytes() for p in settings.processed_dir.iterdir()} == before
    finally:
        app.dependency_overrides.pop(get_retrieval_service, None)
