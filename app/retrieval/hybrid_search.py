"""Reciprocal rank fusion; raw cosine and BM25 scores are never added."""

import logging
from contextlib import contextmanager
from time import perf_counter

from app.config.settings import get_settings
from app.database.connection import get_connection
from app.retrieval.bm25_search import BM25Retriever
from app.retrieval.models import EvidenceResult, HybridResult, RetrievalRequest
from app.retrieval.vector_search import VectorRetriever

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(vector_results, bm25_results, top_k=10, *, rank_constant=60):
    """Fuse unique candidates by rank with deterministic tie-breaking."""
    if type(top_k) is not int or not 1 <= top_k <= 100:
        raise ValueError("top_k must be an integer between 1 and 100")
    if type(rank_constant) is not int or rank_constant < 1:
        raise ValueError("rank_constant must be a positive integer")
    merged = {}
    for source, results in (("vector", vector_results), ("bm25", bm25_results)):
        seen = set()
        for result in results:
            if result.chunk_id in seen:
                continue
            seen.add(result.chunk_id)
            rank = len(seen)
            if result.chunk_id not in merged:
                merged[result.chunk_id] = dict(
                    EvidenceResult.model_validate(result.model_dump()).model_dump(),
                    rrf_score=0.0,
                    hybrid_rank=0,
                    sources=[],
                )
            entry = merged[result.chunk_id]
            entry["rrf_score"] += 1 / (rank_constant + rank)
            entry[source + "_contribution"] = 1 / (rank_constant + rank)
            entry[source + "_rank"] = rank
            entry["sources"].append(source)
            if source == "vector":
                entry["cosine_similarity"] = result.cosine_similarity
            else:
                entry["bm25_score"] = result.bm25_score
    ordered = sorted(merged.values(), key=lambda r: (-r["rrf_score"], r["chunk_id"]))[:top_k]
    return [
        HybridResult(**dict(record, hybrid_rank=rank)) for rank, record in enumerate(ordered, 1)
    ]


class HybridRetriever:
    """Combine lexical and semantic candidates from one guarded corpus generation."""

    def __init__(self, *, embedder=None, connection_factory=get_connection, settings=None):
        self.settings = settings or get_settings()
        self.connection_factory = connection_factory
        self.embedder = embedder
        self.bm25 = BM25Retriever(settings=self.settings, connection_factory=connection_factory)

    def search(self, query, top_k=10, filters=None, *, candidate_k=None):
        """Return bounded ranked evidence for a validated query and its filters."""
        request = RetrievalRequest(
            query=query, top_k=top_k, filters={} if filters is None else filters
        )
        if candidate_k is None:
            candidate_k = max(top_k, self.settings.retrieval_candidate_k)
        candidates = RetrievalRequest(
            query=request.query, top_k=candidate_k, filters=request.filters
        )
        if candidate_k < top_k:
            raise ValueError("candidate_k must be at least top_k")

        with self.connection_factory() as conn:
            started = perf_counter()
            keywords = self.bm25.search_connection(conn, candidates)
            keyword_ms = (perf_counter() - started) * 1000
            started = perf_counter()

            @contextmanager
            def same_connection():
                yield conn

            vectors = VectorRetriever(
                embedder=self.embedder, settings=self.settings, connection_factory=same_connection
            ).search(request.query, candidate_k, request.filters)
            vector_ms = (perf_counter() - started) * 1000
        started = perf_counter()
        results = reciprocal_rank_fusion(
            vectors, keywords, top_k, rank_constant=self.settings.retrieval_rrf_constant
        )
        logger.info(
            "Hybrid query_chars=%d top_k=%d candidate_k=%d bm25_ms=%.1f vector_ms=%.1f fusion_ms=%.1f",
            len(request.query),
            top_k,
            candidate_k,
            keyword_ms,
            vector_ms,
            (perf_counter() - started) * 1000,
        )
        return results
