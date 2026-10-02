"""Exercise ordering, bounded inference, evidence preservation and API failure paths."""

from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config.settings import Settings
from app.main import app
from app.retrieval.embeddings import EmbeddingInputTooLong
from app.retrieval.models import BM25Result, SearchRequest
from app.retrieval.reranking import RerankerUnavailable, RerankingService
from app.retrieval.service import RetrievalService, get_retrieval_service


def evidence(index, content="router drops"):
    return BM25Result(
        chunk_id=index,
        doc_id=str(index),
        chunk_index=0,
        content=content,
        title="Ticket",
        doc_type="resolved_ticket",
        response="private response",
        resolution="simulated fix",
        outcome_status="simulated_resolved",
        metadata={"queue": "support"},
        bm25_score=10 - index,
        bm25_rank=index,
    )


def reranker(scores):
    model = Mock()
    model.predict.return_value = np.array(scores)
    model.tokenizer.encode.side_effect = lambda a, b=None, **kwargs: list(
        range(len(a.split()) + (len(b.split()) if b else 0) + 3)
    )
    return RerankingService(settings=Settings(_env_file=None), model=model)


def test_rank_ties_and_evidence_preserved_without_mutation():
    candidates = [evidence(i) for i in (1, 2, 3)]
    ranker = reranker([-3, 4, 4])
    result = ranker.rerank("drops", candidates, 3)
    assert [r.chunk_id for r in result] == [2, 3, 1]
    assert [r.rerank_rank for r in result] == [1, 2, 3]
    assert result[0].bm25_rank == 2 and result[0].outcome_status == "simulated_resolved"
    assert result[0].resolution == candidates[1].resolution
    assert all(r.rerank_score is None for r in candidates)
    assert ranker.model.predict.call_args.args[0] == [("drops", r.content) for r in candidates]


@pytest.mark.parametrize("scores", [[float("nan")], [float("inf")], [], [[1, 2]]])
def test_invalid_model_output_fails_explicitly(scores):
    with pytest.raises(RerankerUnavailable):
        reranker(scores).rerank("drops", [evidence(1)], 1)


def test_empty_pool_skips_model_inference():
    ranker = reranker([])
    assert ranker.rerank("drops", [], 1) == []
    ranker.model.predict.assert_not_called()


def test_long_query_rejected_and_long_evidence_truncation_disclosed():
    ranker = reranker([1])
    with pytest.raises(EmbeddingInputTooLong):
        ranker.rerank("x " * 260, [evidence(1)], 1)
    ranker.model.predict.assert_not_called()
    candidate = evidence(1, "evidence " * 600)
    result = ranker.rerank("drops", [candidate], 1)[0]
    assert result.rerank_input_truncated and result.content == candidate.content


def test_duplicate_chunks_and_oversized_pool_rejected():
    ranker = reranker([1])
    with pytest.raises(ValueError):
        ranker.rerank("drops", [evidence(1), evidence(1)], 1)
    with pytest.raises(ValueError):
        ranker.rerank("drops", [evidence(i) for i in range(51)], 5)


@pytest.mark.parametrize(
    "options",
    [
        {"rerank_k": 20},
        {"rerank": "true"},
        {"rerank": True, "top_k": 51},
        {"rerank": True, "rerank_k": 4},
        {"rerank": True, "rerank_k": 51},
        {"rerank": True, "candidate_k": 10},
    ],
)
def test_invalid_reranking_bounds(options):
    with pytest.raises(ValidationError):
        SearchRequest(query="drops", **options)


@pytest.mark.parametrize("mode", ["bm25", "vector", "hybrid"])
def test_service_fetches_pool_then_reranks_and_preserves_filters(mode):
    retriever = Mock()
    retriever.search.return_value = [evidence(1), evidence(2)]
    service = RetrievalService(
        bm25=retriever, vector=retriever, hybrid=retriever, reranker=reranker([0, 1])
    )
    request = SearchRequest(
        query="drops", top_k=1, mode=mode, rerank=True, rerank_k=10, filters={"queue": "support"}
    )
    result = service.search(request)
    assert retriever.search.call_args.args[1] == 10
    assert retriever.search.call_args.args[2] == request.filters
    assert result.reranked and result.reranked_candidates == 2
    assert result.results[0].chunk_id == 2


def test_reranking_failure_is_503_not_silent_fallback():
    service = Mock()
    service.search.side_effect = RerankerUnavailable("private model error")
    app.dependency_overrides[get_retrieval_service] = lambda: service
    try:
        response = TestClient(app).post("/api/v1/retrieve", json={"query": "drops", "rerank": True})
        assert response.status_code == 503 and "private" not in response.text
    finally:
        app.dependency_overrides.pop(get_retrieval_service, None)


def test_evaluation_does_not_look_beyond_top_k_to_skip_duplicates():
    from scripts.evaluate_reranking import relevance

    pool = [evidence(1), evidence(1), evidence(2)]
    assert relevance(pool, {"2"}, 2) == {"hit": 0, "reciprocal_rank": 0.0}
    assert relevance(pool, {"2"}, 3)["reciprocal_rank"] == pytest.approx(1 / 3)
