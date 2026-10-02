"""Adversarial regressions for ranking, request validation and dataset boundaries."""

import copy
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from app.ingestion.synthetic import build_dataset, load_scenarios, validate_dataset
from app.retrieval.bm25_search import BM25Retriever
from app.retrieval.hybrid_search import HybridRetriever, reciprocal_rank_fusion
from app.retrieval.models import VectorResult
from app.retrieval.vector_search import VectorRetriever


@pytest.mark.parametrize("retriever_class", [BM25Retriever, VectorRetriever, HybridRetriever])
@pytest.mark.parametrize("filters", [[], "", False, 0])
def test_invalid_empty_filters_never_reach_database(retriever_class, filters):
    connection = Mock()
    retriever = retriever_class(connection_factory=connection)
    with pytest.raises(ValidationError):
        retriever.search("router", filters=filters)
    connection.assert_not_called()


def test_duplicate_candidate_cannot_penalize_the_next_unique_candidate():
    def candidate(identifier):
        return VectorResult(
            chunk_id=identifier,
            doc_id=str(identifier),
            chunk_index=0,
            content="Router power fault",
            title="Power",
            doc_type="knowledge_base",
            response=None,
            resolution=None,
            outcome_status="unknown",
            metadata={},
            cosine_distance=0.2,
            cosine_similarity=0.8,
            vector_rank=1,
        )

    first, second = candidate(1), candidate(2)
    results = reciprocal_rank_fusion([first, first, second], [])
    assert results[1].vector_rank == 2
    assert results[1].rrf_score == pytest.approx(1 / 62)


def test_catalog_reordering_does_not_change_splits_or_document_ids():
    scenarios = load_scenarios()
    assert build_dataset(scenarios) == build_dataset(list(reversed(scenarios)))


@pytest.mark.parametrize(
    "corruption", ["missing_reference", "split_lie", "positive_as_negative", "ticket_reference"]
)
def test_dataset_rejects_silent_training_corruption(corruption):
    dataset = copy.deepcopy(build_dataset(load_scenarios()))
    if corruption == "missing_reference":
        dataset["test"][0]["relevant_kb_ids"] = ["missing"]
    elif corruption == "split_lie":
        dataset["dev"][0]["split"] = "train"
    elif corruption == "positive_as_negative":
        dataset["train"][0]["contrast_candidate_ids"] = dataset["train"][0]["relevant_kb_ids"]
    else:
        dataset["tickets"][0]["metadata"]["kb_refs"] = ["missing"]
    with pytest.raises(ValueError):
        validate_dataset(dataset)
