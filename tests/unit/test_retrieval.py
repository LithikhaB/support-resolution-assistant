"""Core vector, lexical, fusion and reranking checks."""

from contextlib import contextmanager
from unittest.mock import Mock

import numpy as np
import pytest

from app.config.settings import Settings
from app.retrieval.bm25_search import BM25Index
from app.retrieval.hybrid_search import reciprocal_rank_fusion
from app.retrieval.lexical_search import build_lexical_sql
from app.retrieval.models import BM25Result, RetrievalFilters, RetrievalRequest, VectorResult
from app.retrieval.reranking import RerankingService
from app.retrieval.vector_search import VectorRetriever, build_search_sql


def test_filter_values_are_parameters():
    attack = "x' OR 1=1 --"
    sql, params = build_search_sql(RetrievalFilters(queue=attack), "[1]", 5)
    assert attack not in sql and attack in params
    assert "MATERIALIZED" in sql
    assert "c.content" not in sql.split("), winners")[0]
    assert params[-1] == 5


def setup_retriever(state="ready", lock=True):
    settings = Settings(_env_file=None)
    config = {
        "model": settings.embedding_model,
        "tokenizer_requested_revision": settings.tokenizer_revision,
        "dimension": 384,
        "normalized": True,
    }
    conn = Mock()

    def execute(sql, params=None):
        result = Mock()
        if "advisory" in sql:
            result.fetchone.return_value = (lock,)
        elif "retrieval_index_state" in sql:
            result.fetchone.return_value = (state, config)
        else:
            result.fetchall.return_value = [
                (
                    1,
                    "doc",
                    0,
                    "complaint",
                    "Title",
                    "historical_response",
                    "Waiting for details",
                    None,
                    "unknown",
                    {},
                    0.25,
                )
            ]
        return result

    conn.execute.side_effect = execute

    @contextmanager
    def connection():
        yield conn

    embedder = Mock()
    embedder.embed_query.return_value = [1.0] + [0.0] * 383
    return (
        VectorRetriever(embedder=embedder, connection_factory=connection, settings=settings),
        embedder,
        config,
    )


def test_results_preserve_evidence_and_distance_semantics():
    retriever, embedder, _ = setup_retriever()
    result = retriever.search(" complaint ")[0]
    assert result.cosine_distance == 0.25 and result.cosine_similarity == 0.75
    assert result.response == "Waiting for details" and result.resolution is None
    assert result.outcome_status == "unknown" and result.vector_rank == 1
    embedder.embed_query.assert_called_once_with("complaint")


def test_query_and_metadata_never_become_sql():
    marker = "x' OR 1=1 --"
    query, parameters = build_lexical_sql(
        RetrievalRequest(query=marker, filters={"queue": marker}, top_k=3)
    )
    assert marker not in query
    assert marker in parameters
    assert "plainto_tsquery" in query and "search_vector @@" in query
    assert parameters[-1] == 3


def record(chunk_id, content, queue="technical_support"):
    return dict(
        chunk_id=chunk_id,
        doc_id=str(chunk_id),
        chunk_index=0,
        content=content,
        title="Ticket",
        doc_type="historical_response",
        response="Ask for logs",
        resolution=None,
        outcome_status="unknown",
        metadata={"queue": queue},
        intent=None,
        severity=None,
        ticket_type="problem",
    )


def test_rrf_agreement_ranks_and_deduplication():
    keywords = BM25Index([record(1, "router"), record(2, "router")]).search(
        RetrievalRequest(query="router")
    )
    vectors = [
        VectorResult(
            **record(i, "router"), cosine_distance=0.25, cosine_similarity=0.75, vector_rank=rank
        )
        for rank, i in enumerate((2, 3), 1)
    ]
    results = reciprocal_rank_fusion(vectors + vectors, keywords, 3)
    assert [r.chunk_id for r in results] == [2, 1, 3]
    assert results[0].rrf_score == pytest.approx(1 / 61 + 1 / 62)
    assert results[0].vector_contribution == pytest.approx(1 / 61)
    assert results[0].bm25_contribution == pytest.approx(1 / 62)
    assert results[1].vector_contribution == 0
    assert results[0].sources == ["vector", "bm25"]
    assert results[0].vector_rank == 1 and results[0].bm25_rank == 2
    assert results[1].vector_rank is None
    assert reciprocal_rank_fusion([], []) == []
    assert len(reciprocal_rank_fusion([], keywords, 1)) == 1


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

    from app.retrieval.cache import TTLCache
    from app.retrieval.models import SearchRequest
    from app.retrieval.service import RetrievalService
    from app.retrieval.vector_search import RetrievalUnavailable

    clock = Mock(return_value=0)
    cache = TTLCache(capacity=1, seconds=5, clock=clock)
    cache.put("a", [1])
    cache.get("a").append(2)
    assert cache.get("a") == [1]
    cache.put("b", [2])
    assert cache.get("a") is None
    clock.return_value = 6
    assert cache.get("b") is None
    version = Mock(return_value=("corpus1", "embedding1"))
    backend = Mock()
    backend.search.side_effect = lambda *args, **kwargs: [evidence(1)]
    search = RetrievalService(hybrid=backend, revision=version)
    request = SearchRequest(query="drops", rerank=False)
    search.search(request).results[0].resolution = "mutated"
    assert search.search(request).results[0].resolution == "simulated fix"
    assert backend.search.call_count == 1
    version.return_value = ("corpus2", "embedding1")
    search.search(request)
    assert backend.search.call_count == 2
    version.side_effect = RetrievalUnavailable("index_update_in_progress")
    with pytest.raises(RetrievalUnavailable):
        search.search(request)


def test_reranker_pool_runs_independent_instances_and_returns_them_after_errors():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from types import SimpleNamespace

    from app.retrieval.reranking import RerankerPool

    barrier = Barrier(2)

    def rank(query, candidates, limit):
        barrier.wait(timeout=2)
        return query

    models = [SimpleNamespace(settings=None, model=None, rerank=rank) for _ in range(2)]
    pool = RerankerPool(models)
    with ThreadPoolExecutor(max_workers=2) as executor:
        assert set(executor.map(lambda query: pool.rerank(query, [], 1), ("first", "second"))) == {
            "first",
            "second",
        }
    assert pool.available.qsize() == 2

    def fail(*args):
        raise RuntimeError("model failed")

    for model in models:
        model.rerank = fail
    with pytest.raises(RuntimeError, match="model failed"):
        pool.rerank("query", [], 1)
    assert pool.available.qsize() == 2
