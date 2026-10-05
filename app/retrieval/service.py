"""Application-facing retrieval service; model and lexical cache live per process."""

import logging
from functools import lru_cache
from threading import Lock
from time import perf_counter

from app.config.settings import get_settings
from app.ingestion.artifacts import digest
from app.retrieval.cache import TTLCache, published_revision
from app.retrieval.diversity import diverse_results
from app.retrieval.hybrid_search import HybridRetriever
from app.retrieval.models import SearchRequest, SearchResponse
from app.retrieval.reranking import get_reranking_service
from app.retrieval.vector_search import VectorRetriever

logger = logging.getLogger(__name__)
_factory_lock = Lock()


class RetrievalService:
    """Dispatch validated application requests to the selected retrieval strategy."""

    def __init__(
        self, *, hybrid=None, vector=None, bm25=None, reranker=None, revision=None, settings=None
    ):
        self.reranker = reranker
        self.hybrid = hybrid if hybrid is not None else HybridRetriever()
        self.vector = vector if vector is not None else VectorRetriever()
        self.bm25 = bm25 if bm25 is not None else self.hybrid.bm25
        self.revision = revision or published_revision
        self.cache = TTLCache(seconds=(settings or get_settings()).computation_cache_seconds)

    def by_ids(self, identifiers):
        """Load current source records referenced by a reviewed outcome, without broad search."""
        from app.database.connection import get_connection
        from app.retrieval.models import EvidenceResult
        from app.retrieval.vector_search import RetrievalUnavailable

        self.revision()
        with get_connection() as conn:
            if not conn.execute("SELECT pg_try_advisory_xact_lock_shared(8041,2)").fetchone()[0]:
                raise RetrievalUnavailable("index_update_in_progress")
            rows = conn.execute(
                "SELECT c.chunk_id,c.doc_id,c.chunk_index,c.content,d.title,d.doc_type,"
                "d.response,d.resolution,d.outcome_status,d.metadata,d.body "
                "FROM chunks c JOIN documents d USING(doc_id) "
                "WHERE d.doc_id=ANY(%s) AND d.doc_type='knowledge_base' ORDER BY c.chunk_id",
                (list(identifiers),),
            ).fetchall()
        return [
            EvidenceResult(
                **dict(
                    zip(
                        (
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
                            "evidence_content",
                        ),
                        row,
                        strict=True,
                    )
                )
            )
            for row in rows
        ]

    def search(self, request: SearchRequest) -> SearchResponse:
        """Return bounded ranked evidence for a validated query and its filters."""
        started = perf_counter()
        key = digest([self.revision(), request.model_dump(mode="json")])
        cached = self.cache.get(key)
        if cached is not None:
            cached.elapsed_ms = (perf_counter() - started) * 1000
            return cached
        retriever = {"hybrid": self.hybrid, "vector": self.vector, "bm25": self.bm25}[request.mode]
        options = {"candidate_k": request.candidate_k} if request.mode == "hybrid" else {}
        results = retriever.search(
            request.query, request.retrieval_depth, request.filters, **options
        )
        retrieved_count = len(results)
        if request.diversify:
            results = diverse_results(results, request.ranking_depth)
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
        response = SearchResponse(
            mode=request.mode,
            results=results,
            elapsed_ms=elapsed,
            diversified=request.diversify,
            retrieved_candidates=retrieved_count,
            **reranking,
        )
        self.cache.put(key, response)
        return response


@lru_cache(maxsize=1)
def _cached_service() -> RetrievalService:
    return RetrievalService()


def get_retrieval_service() -> RetrievalService:
    """Return the reusable retrieval orchestrator without loading models eagerly."""
    with _factory_lock:
        return _cached_service()
