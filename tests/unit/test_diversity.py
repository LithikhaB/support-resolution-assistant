"""Test family diversity without altering source evidence or excluding independent cases."""

from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from app.retrieval.diversity import diverse_results
from app.retrieval.models import BM25Result, SearchRequest
from app.retrieval.service import RetrievalService


def record(index, family=None, source="provider", doc_type="resolved_ticket"):
    return BM25Result(
        chunk_id=index,
        doc_id=str(index),
        chunk_index=0,
        content="broadband drops",
        title="Ticket",
        doc_type=doc_type,
        response=None,
        resolution="example",
        outcome_status="simulated_resolved",
        metadata={"is_synthetic": True, "scenario_family": family, "source": source},
        bm25_score=1,
        bm25_rank=index,
    )


def test_siblings_collapse_but_other_families_and_kb_remain():
    items = [
        record(1, "A"),
        record(2, "A"),
        record(3, "A", doc_type="knowledge_base"),
        record(4, "B"),
    ]
    result = diverse_results(items, 5)
    assert [r.chunk_id for r in result] == [1, 3, 4]
    assert result[1].bm25_rank == 3 and result[0] is items[0]


def test_providers_real_cases_and_missing_family_are_not_conflated():
    first = record(1, "A", "one")
    second = record(2, "A", "two")
    third = record(3, "A", "one")
    third.metadata["is_synthetic"] = False
    assert len(diverse_results([first, second, third, record(4), record(5)], 5)) == 5


def test_multiple_chunks_of_document_collapse_without_family_metadata():
    first = record(1)
    second = first.model_copy(update={"chunk_id": 2, "chunk_index": 1})
    assert diverse_results([first, second], 5) == [first]


def test_service_overfetches_without_leaking_duplicates_or_changing_filters():
    retriever = Mock()
    retriever.search.return_value = [record(1, "A"), record(2, "A"), record(3, "B")]
    service = RetrievalService(hybrid=retriever, vector=retriever, bm25=retriever)
    response = service.search(
        SearchRequest(query="drops", top_k=2, diversify=True, filters={"queue": "support"})
    )
    assert retriever.search.call_args.args[1] == 50
    assert retriever.search.call_args.args[2].queue == "support"
    assert [r.chunk_id for r in response.results] == [1, 3]
    assert response.diversified and response.retrieved_candidates == 3


@pytest.mark.parametrize(
    "options",
    [
        {"diversify": True, "candidate_k": 10},
        {"diversify": "true"},
        {"diversify": True, "rerank": True, "rerank_k": 4},
    ],
)
def test_invalid_diversity_options(options):
    with pytest.raises(ValidationError):
        SearchRequest(query="drops", **options)


def test_reranking_pool_is_bounded_after_diversity():
    retriever = Mock()
    retriever.search.return_value = [record(i, str(i)) for i in range(50)]
    ranker = Mock()
    ranker.rerank.side_effect = lambda query, candidates, top_k: candidates[:top_k]
    ranker.settings.reranker_model = "test"
    ranker.settings.reranker_revision = "revision"
    result = RetrievalService(hybrid=retriever, reranker=ranker).search(
        SearchRequest(query="drops", diversify=True, rerank=True, rerank_k=20)
    )
    assert len(ranker.rerank.call_args.args[1]) == 20
    assert result.reranked_candidates == 20 and len(result.results) == 5
