"""Live ingestion stays disabled without a key and preserves active artifacts on failure."""

from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from tokenizers import Tokenizer, models, pre_tokenizers

from app.api import ingest as api
from app.config.settings import Settings
from app.ingestion import live
from app.ingestion.artifacts import atomic_write, file_sha256, write_json
from app.ingestion.chunking import chunk_corpus
from app.ingestion.schema import SupportDocument
from app.main import app


def article(identifier="new", **changes):
    return {
        "doc_id": identifier,
        "doc_type": "knowledge_base",
        "title": "Guide",
        "body": "DNS domain lookup support investigation.",
        "intent": "broadband_dns",
        **changes,
    }


class Connection:
    def __init__(self, state):
        self.state = state

    def execute(self, *args):
        return SimpleNamespace(fetchone=lambda: ("ready", *self.state))

    @contextmanager
    def transaction(self):
        before = deepcopy(self.state)
        try:
            yield
        except BaseException:
            self.state = before
            raise


@pytest.fixture
def environment(tmp_path, monkeypatch):
    settings = Settings(
        _env_file=None, ingest_admin_key="test-admin", corpus_dir=tmp_path / "corpus"
    )
    active = settings.processed_dir
    active.mkdir(parents=True)
    original = SupportDocument.model_validate(article("old", intent="broadband_outage"))
    atomic_write(active / "documents.jsonl", [original.model_dump_json() + "\n"])
    write_json(active / "manifest.json", {"output_sha256": file_sha256(active / "documents.jsonl")})
    tok = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    token_path = tmp_path / "tokenizer.json"
    tok.save(str(token_path))
    monkeypatch.setattr("app.ingestion.chunking.load_tokenizer", lambda *args: (tok, token_path))
    chunk_corpus(settings)
    write_json(active / "indexing.manifest.json", {"old": True})
    conn = Connection(
        (
            file_sha256(active / "documents.jsonl"),
            file_sha256(active / "chunks.jsonl"),
            "config",
            "old",
        )
    )

    @contextmanager
    def writer(_):
        yield conn

    monkeypatch.setattr(live, "writer", writer)
    monkeypatch.setattr(
        live,
        "get_understanding_service",
        lambda: SimpleNamespace(
            classifier=SimpleNamespace(artifact=SimpleNamespace(classes=["broadband_outage"]))
        ),
    )
    monkeypatch.setattr(api, "get_settings", lambda: settings)
    monkeypatch.setattr(live, "get_settings", lambda: settings)

    def index(candidate):
        write_json(candidate.processed_dir / "indexing.manifest.json", {"new": True})
        conn.state = (
            file_sha256(candidate.processed_dir / "documents.jsonl"),
            file_sha256(candidate.processed_dir / "chunks.jsonl"),
            "config",
            "new",
        )
        return {"embeddings_generated": 1}

    indexer = Mock(side_effect=index)
    monkeypatch.setattr(live, "index_corpus", indexer)
    return settings, conn, indexer


def test_ingest_disabled_and_wrong_key_return_403(environment, monkeypatch):
    settings, _, indexer = environment
    client = TestClient(app)
    assert (
        client.post(
            "/api/v1/ingest", json={"documents": [article()]}, headers={"X-Admin-Key": "wrong"}
        ).status_code
        == 403
    )
    monkeypatch.setattr(
        api,
        "get_settings",
        lambda: settings.model_copy(
            update={"ingest_admin_key": Settings(_env_file=None).ingest_admin_key}
        ),
    )
    assert (
        client.post(
            "/api/v1/ingest", json={"documents": [article()]}, headers={"X-Admin-Key": "test-admin"}
        ).status_code
        == 403
    )
    indexer.assert_not_called()


@pytest.mark.parametrize(
    "documents",
    [
        [article(metadata={"split": "dev"})],
        [article(metadata={"split": "test"})],
        [article(metadata={"kb_refs": ["missing"]})],
        [article(), article()],
        [article(str(i)) for i in range(21)],
    ],
)
def test_invalid_ingest_leaves_files_untouched(environment, documents):
    settings, _, indexer = environment
    before = {p.name: p.read_bytes() for p in settings.processed_dir.iterdir()}
    response = TestClient(app).post(
        "/api/v1/ingest", json={"documents": documents}, headers={"X-Admin-Key": "test-admin"}
    )
    assert response.status_code == 422
    assert {p.name: p.read_bytes() for p in settings.processed_dir.iterdir()} == before
    indexer.assert_not_called()


def test_success_indexes_once_and_marks_new_category(environment):
    settings, _, indexer = environment
    response = TestClient(app).post(
        "/api/v1/ingest", json={"documents": [article()]}, headers={"X-Admin-Key": "test-admin"}
    )
    assert response.status_code == 200
    result = response.json()
    assert result["added"] == 1 and result["updated"] == 0 and result["embeddings_generated"] == 1
    assert result["new_categories"] == ["broadband_dns"] and result["classifier_retrain_required"]
    assert result["revision"][-1] == "new"
    assert "python -m scripts train" in result["category_prediction_note"]
    indexer.assert_called_once()
    assert "new" in (settings.processed_dir / "documents.jsonl").read_text()


@pytest.mark.parametrize("failure", ["index", "publish", "commit"])
def test_failed_ingest_rolls_back_files_and_index(environment, monkeypatch, failure):
    settings, conn, indexer = environment
    before = {p.name: p.read_bytes() for p in settings.processed_dir.iterdir()}
    old_state = conn.state
    if failure == "index":
        indexer.side_effect = RuntimeError("private failure text")
    elif failure == "publish":

        def fail(staged, active):
            atomic_write(active / "documents.jsonl", ["partially published"])
            raise OSError("private failure text")

        monkeypatch.setattr(live, "publish_artifacts", fail)
    else:

        @contextmanager
        def fail_commit():
            try:
                yield
                raise OSError("private failure text")
            finally:
                conn.state = old_state

        monkeypatch.setattr(conn, "transaction", fail_commit)
    response = TestClient(app).post(
        "/api/v1/ingest", json={"documents": [article()]}, headers={"X-Admin-Key": "test-admin"}
    )
    assert response.status_code == 503 and "private failure" not in response.text
    assert {p.name: p.read_bytes() for p in settings.processed_dir.iterdir()} == before
    assert conn.state == old_state


def test_ingest_metrics_and_by_id_replace(environment):
    from app.monitoring.metrics import _ingest_counts, prometheus_metrics

    _, _, indexer = environment
    before = dict(_ingest_counts)
    client = TestClient(app)
    rejected = client.post("/api/v1/ingest", json={"documents": [article()]})
    assert rejected.status_code == 403
    result = client.post(
        "/api/v1/ingest",
        json={"documents": [article("old", intent="broadband_outage")]},
        headers={"X-Admin-Key": "test-admin"},
    )
    assert result.status_code == 200
    assert result.json()["added"] == 0 and result.json()["updated"] == 1
    assert result.json()["new_categories"] == []
    assert not result.json()["classifier_retrain_required"]
    indexer.assert_called_once()
    assert _ingest_counts["success"] == before.get("success", 0) + 1
    assert _ingest_counts["failure"] == before.get("failure", 0) + 1
    output = prometheus_metrics().body.decode()
    assert 'ingest_total{outcome="success"}' in output
    assert "ingest_documents_added_total" in output
