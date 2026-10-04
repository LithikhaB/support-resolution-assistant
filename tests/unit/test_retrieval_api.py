from unittest.mock import Mock

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.retrieval.embeddings import EmbeddingInputTooLong
from app.retrieval.models import SearchRequest, SearchResponse
from app.retrieval.service import RetrievalService, get_retrieval_service
from app.retrieval.vector_search import RetrievalUnavailable


@pytest.fixture
def api():
    service = Mock()
    service.search.return_value = SearchResponse(mode="hybrid", results=[], elapsed_ms=1)
    app.dependency_overrides[get_retrieval_service] = lambda: service
    try:
        yield TestClient(app), service
    finally:
        app.dependency_overrides.pop(get_retrieval_service, None)


def test_retrieve_default_and_empty_result(api):
    client, service = api
    response = client.post("/api/v1/retrieve", json={"query": " router "})
    assert response.status_code == 200
    assert response.json() == SearchResponse(mode="hybrid", results=[], elapsed_ms=1).model_dump()
    request = service.search.call_args.args[0]
    assert request.query == "router" and request.top_k == 5


@pytest.mark.parametrize(
    "values",
    [
        {"query": ""},
        {"query": " "},
        {"query": "a" * 10001},
        {"query": 5},
        {"top_k": 101},
        {"top_k": True},
        {"top_k": 0},
        {"mode": "rag"},
        {"candidate_k": 3},
        {"candidate_k": 101},
        {"candidate_k": True},
        {"mode": "bm25", "candidate_k": 50},
        {"filters": {"unknown": "value"}},
    ],
)
def test_api_rejects_invalid_requests_without_retrieval(api, values):
    client, service = api
    response = client.post("/api/v1/retrieve", json={"query": "router", **values})
    assert response.status_code == 422
    service.search.assert_not_called()


@pytest.mark.parametrize(
    "error,status",
    [
        (EmbeddingInputTooLong("private input"), 422),
        (psycopg.errors.QueryCanceled("private query"), 504),
        (psycopg.OperationalError("private password"), 503),
        (psycopg.errors.UndefinedTable("private schema"), 503),
        (RetrievalUnavailable("private configuration"), 503),
        (OSError("private model path"), 503),
    ],
)
def test_failures_are_actionable_and_sanitized(api, error, status, caplog):
    client, service = api
    service.search.side_effect = error
    response = client.post("/api/v1/retrieve", json={"query": "secret complaint"})
    assert response.status_code == status
    assert "private" not in response.text + caplog.text
    assert "secret complaint" not in caplog.text
    assert client.get("/api/v1/health").status_code == 200


@pytest.mark.parametrize("mode", ["bm25", "vector", "hybrid"])
def test_service_routes_and_logs_without_query(mode, caplog):
    retrievers = {name: Mock() for name in ("bm25", "vector", "hybrid")}
    for retriever in retrievers.values():
        retriever.search.return_value = []
    service = RetrievalService(**retrievers)
    with caplog.at_level("INFO"):
        response = service.search(SearchRequest(query="private customer text", mode=mode))
    assert response.mode == mode and response.elapsed_ms >= 0
    retrievers[mode].search.assert_called_once()
    assert "private customer text" not in caplog.text
    for name, retriever in retrievers.items():
        if name != mode:
            retriever.search.assert_not_called()
