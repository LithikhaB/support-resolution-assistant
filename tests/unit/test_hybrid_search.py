from contextlib import contextmanager
from math import log
from unittest.mock import Mock

import pytest

from app.retrieval.bm25_search import BM25Index, BM25Retriever, tokenize
from app.retrieval.hybrid_search import HybridRetriever, reciprocal_rank_fusion
from app.retrieval.models import RetrievalRequest, VectorResult
from app.retrieval.vector_search import RetrievalUnavailable


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


def test_bm25_formula_and_no_response_leakage():
    index = BM25Index([record(1, "router router"), record(2, "printer printer")])
    results = index.search(RetrievalRequest(query="ROUTER"))
    assert [r.chunk_id for r in results] == [1]
    assert results[0].bm25_score == pytest.approx(log(2) * 5 / 3.5)
    assert results[0].resolution is None and results[0].outcome_status == "unknown"
    assert index.search(RetrievalRequest(query="logs")) == []


def test_bm25_filters_ties_empty_and_identifiers():
    index = BM25Index([record(2, "ERR-42 router"), record(1, "ERR-42 router", "billing")])
    assert tokenize("ERR-42 Django 3.2") == ["err-42", "django", "3.2"]
    assert [r.chunk_id for r in index.search(RetrievalRequest(query="err-42"))] == [1, 2]
    filtered = RetrievalRequest(
        query="err-42", filters={"queue": "technical_support", "ticket_type": "problem"}
    )
    assert [r.chunk_id for r in index.search(filtered)] == [2]
    for query in ("unseen", "!!!"):
        assert index.search(RetrievalRequest(query=query)) == []
    assert BM25Index([]).search(RetrievalRequest(query="router")) == []
    assert BM25Index([record(1, "")]).search(RetrievalRequest(query="router")) == []


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


def test_hybrid_rejects_insufficient_candidates_before_database():
    connection = Mock()
    with pytest.raises(ValueError):
        HybridRetriever(connection_factory=connection).search("router", 10, candidate_k=5)
    connection.assert_not_called()


def test_bm25_cache_refresh_and_not_ready():
    conn = Mock()
    state = ["ready", "config", "source", "chunks", "time1"]
    fields = (
        "chunk_id",
        "doc_id",
        "chunk_index",
        "content",
        "title",
        "doc_type",
        "response",
        "resolution",
        "outcome_status",
        "metadata",
        "intent",
        "severity",
        "ticket_type",
    )
    loads = []

    def execute(sql, params=None):
        result = Mock()
        if "advisory" in sql:
            result.fetchone.return_value = (True,)
        elif "retrieval_index_state" in sql:
            result.fetchone.return_value = tuple(state)
        elif "FROM chunks" in sql:
            loads.append(sql)
            result.fetchall.return_value = [tuple(record(1, "router")[k] for k in fields)]
        return result

    conn.execute.side_effect = execute

    @contextmanager
    def connection():
        yield conn

    retriever = BM25Retriever(connection_factory=connection)
    assert retriever.search("router")[0].chunk_id == 1
    retriever.search("router")
    assert len(loads) == 1
    state[-1] = "time2"
    retriever.search("router")
    assert len(loads) == 2
    state[0] = "indexing"
    with pytest.raises(RetrievalUnavailable):
        retriever.search("router")
