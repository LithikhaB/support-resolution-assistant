"""Combine complaint understanding, diverse KB retrieval and conditional drafting."""

import logging
from functools import lru_cache
from time import perf_counter

from app.resolution.applicability import evidence_query
from app.resolution.customer import physical_damage
from app.resolution.drafting import draft_resolution
from app.resolution.evidence import supported_scopes
from app.resolution.validation import finalize_resolution
from app.retrieval.embeddings import EmbeddingInputTooLong
from app.retrieval.models import RetrievalFilters, SearchRequest
from app.retrieval.service import get_retrieval_service
from app.understanding.models import AnalyzeRequest
from app.understanding.service import get_understanding_service

logger = logging.getLogger(__name__)


class ResolutionService:
    """Keep infrastructure failures distinct from an honest lack of applicable evidence."""

    def __init__(self, *, understanding=None, retrieval=None):
        self.understanding = understanding
        self.retrieval = retrieval

    def resolve(self, request, *, analysis=None):
        """Generate a local draft without executing repairs or inferring diagnostic findings."""
        started = perf_counter()
        if analysis is None:
            understanding = self.understanding or get_understanding_service()
            analysis = understanding.analyze(AnalyzeRequest(query=request.query))
        evidence = []
        if (
            analysis.scope_status != "unsupported"
            and supported_scopes(analysis)
            and not physical_damage(analysis)
        ):
            retrieval = self.retrieval or get_retrieval_service()
            filters = RetrievalFilters.model_validate(
                {**request.filters.model_dump(mode="json"), "doc_type": "knowledge_base"}
            )
            search = SearchRequest(
                query=evidence_query(request.query, analysis),
                top_k=20,
                diversify=True,
                rerank=request.rerank,
                filters=filters,
            )
            try:
                evidence = retrieval.search(search).results
            except EmbeddingInputTooLong:
                if search.query == request.query:
                    raise
                evidence = retrieval.search(
                    search.model_copy(update={"query": request.query})
                ).results
        result = draft_resolution(analysis, evidence, max_sources=request.max_sources)
        result = finalize_resolution(result, evidence, analysis)
        result.elapsed_ms = (perf_counter() - started) * 1000
        logger.info(
            "Resolution query_chars=%d status=%s sources=%d elapsed_ms=%.1f",
            len(request.query),
            result.status,
            len(result.sources),
            result.elapsed_ms,
        )
        return result


@lru_cache(maxsize=1)
def get_resolution_service():
    """Reuse the lightweight orchestrator while dependencies retain their own lazy caches."""
    return ResolutionService()
