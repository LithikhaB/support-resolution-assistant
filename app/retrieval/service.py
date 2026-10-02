"""Application-facing retrieval service; model and lexical cache live per process."""

import logging
from functools import lru_cache
from threading import Lock
from time import perf_counter

from app.retrieval.hybrid_search import HybridRetriever
from app.retrieval.models import SearchRequest, SearchResponse
from app.retrieval.reranking import get_reranking_service
from app.retrieval.vector_search import VectorRetriever

logger = logging.getLogger(__name__)
_factory_lock = Lock()


class RetrievalService:
    """Dispatch validated application requests to the selected retrieval strategy."""

    def __init__(self, *, hybrid=None, vector=None, bm25=None, reranker=None):
        self.reranker = reranker
        self.hybrid = hybrid if hybrid is not None else HybridRetriever()
        self.vector = vector if vector is not None else VectorRetriever()
        self.bm25 = bm25 if bm25 is not None else self.hybrid.bm25

    def search(self, request: SearchRequest) -> SearchResponse:
        """Return bounded ranked evidence for a validated query and its filters."""
        started = perf_counter()
        retriever = {"hybrid": self.hybrid, "vector": self.vector, "bm25": self.bm25}[request.mode]
        options = {"candidate_k": request.candidate_k} if request.mode == "hybrid" else {}
        results = retriever.search(
            request.query, request.retrieval_depth, request.filters, **options
        )
        reranking = {}
        if request.rerank and results:
            reranker = self.reranker if self.reranker is not None else get_reranking_service()
            count = len(results)
            results = reranker.rerank(request.query, results, request.top_k)
            reranking = {
                "reranked": True,
                "reranked_candidates": count,
                "reranker_model": reranker.settings.reranker_model,
                "reranker_revision": reranker.settings.reranker_revision,
            }
        elapsed = (perf_counter() - started) * 1000
        logger.info(
            "Retrieval mode=%s query_chars=%d top_k=%d results=%d elapsed_ms=%.1f",
            request.mode,
            len(request.query),
            request.top_k,
            len(results),
            elapsed,
        )
        return SearchResponse(mode=request.mode, results=results, elapsed_ms=elapsed, **reranking)


@lru_cache(maxsize=1)
def _cached_service() -> RetrievalService:
    return RetrievalService()


def get_retrieval_service() -> RetrievalService:
    """Return the reusable retrieval orchestrator without loading models eagerly."""
    with _factory_lock:
        return _cached_service()
