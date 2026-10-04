"""Combine complaint understanding, diverse KB retrieval and conditional drafting."""

import logging
from functools import lru_cache
from time import perf_counter

from app.config.settings import get_settings
from app.ingestion.schema import DocType
from app.llm.client import LanguageUnavailable
from app.llm.providers import ProviderChain, get_language_client
from app.resolution.applicability import evidence_query
from app.resolution.customer import physical_damage
from app.resolution.drafting import draft_resolution
from app.resolution.evidence import supported_scopes
from app.resolution.history import select_history
from app.resolution.language import add_language_draft
from app.resolution.selection import rank_fallback, select_relevant_procedure
from app.resolution.validation import finalize_resolution
from app.retrieval.embeddings import EmbeddingInputTooLong
from app.retrieval.models import RetrievalFilters, SearchRequest
from app.retrieval.service import get_retrieval_service
from app.understanding.models import AnalyzeRequest
from app.understanding.service import get_understanding_service

logger = logging.getLogger(__name__)


class ResolutionService:
    """Keep infrastructure failures distinct from an honest lack of applicable evidence."""

    def __init__(self, *, understanding=None, retrieval=None, settings=None, language=None):
        self.understanding = understanding
        self.retrieval = retrieval
        self.settings = settings or get_settings()
        self.language = language or (
            ProviderChain(self.settings) if settings is not None else get_language_client()
        )

    def resolve(self, request, *, analysis=None):
        """Generate a local draft without executing repairs or inferring diagnostic findings."""
        started = perf_counter()
        if analysis is None:
            understanding = self.understanding or get_understanding_service()
            analysis = understanding.analyze(AnalyzeRequest(query=request.query))
        evidence = []
        history = []
        selection_error = None
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
                search = search.model_copy(update={"query": request.query})
                evidence = retrieval.search(search).results
            if evidence:
                history_search = search.model_copy(update={
                    "filters": filters.model_copy(update={"doc_type": DocType.RESOLVED_TICKET}),
                    "rerank": False,
                })
                history = retrieval.search(history_search).results
                evidence = rank_fallback(evidence)
                if self.settings.llm_enabled:
                    try:
                        evidence = select_relevant_procedure(
                            evidence, request.query, analysis, self.language
                        )
                    except LanguageUnavailable as exc:
                        selection_error = str(exc)
        # One focused procedure per issue; follow-ups can select a different one.
        result = draft_resolution(analysis, evidence, max_sources=1)
        result = finalize_resolution(result, evidence, analysis)
        if selection_error:
            result.limitations.append(
                "Procedure selection unavailable; local ranking retained: " + selection_error
            )
        if result.sources:
            result.historical_cases = select_history(history, result.sources)
        result = add_language_draft(result, request.query, self.settings, self.language)
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
