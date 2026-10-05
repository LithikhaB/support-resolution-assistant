"""Combine complaint understanding, diverse KB retrieval and conditional drafting."""

import json
import logging
from functools import lru_cache
from time import perf_counter

from app.config.settings import get_settings
from app.ingestion.artifacts import digest
from app.ingestion.schema import DocType
from app.llm.client import LanguageUnavailable
from app.llm.disk_cache import DiskCache
from app.llm.providers import ProviderChain, get_language_client
from app.resolution.applicability import applicability_issue, evidence_query
from app.resolution.customer import customer_plan, physical_damage
from app.resolution.drafting import draft_resolution
from app.resolution.evidence import parse_procedure, supported_scopes
from app.resolution.history import select_history
from app.resolution.language import add_language_draft
from app.resolution.memory import history_context
from app.resolution.reuse import SemanticReuse
from app.resolution.selection import rank_fallback, select_relevant_procedure
from app.resolution.validation import finalize_resolution
from app.retrieval.cache import published_revision
from app.retrieval.embeddings import EmbeddingInputTooLong, get_embedding_service
from app.retrieval.models import EvidenceResult, RetrievalFilters, SearchRequest
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
        self.reuse = SemanticReuse(seconds=self.settings.computation_cache_seconds)
        self.language = language or (
            ProviderChain(self.settings) if settings is not None else get_language_client()
        )

    def resolve(self, request, *, analysis=None, trace=None):
        """Generate a local draft without executing repairs or inferring diagnostic findings."""
        started = perf_counter()
        if analysis is None:
            understanding = self.understanding or get_understanding_service()
            analysis = understanding.analyze(AnalyzeRequest(query=request.query))
        evidence = []
        history = []
        selection_error = None
        rerank_disabled = False
        reused = False
        signature = vector = None
        memory = history_context.get()
        cache_key = None
        disk = DiskCache(
            self.settings.data_dir / "cache" / "evidence", self.settings.solution_cache_seconds
        )

        stages = trace if trace is not None else {}
        stages.update(
            {
                "category": analysis.category,
                "scope_status": analysis.scope_status,
                "supported_scopes": sorted(supported_scopes(analysis)),
                "retrieved_kb_ids": [],
                "eligibility": [],
                "selection_method": "local_similarity",
                "selection_error": None,
            }
        )
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
                if self.settings.solution_cache_enabled:
                    revision = published_revision()
                    cache_key = digest(
                        [
                            "evidence-v1",
                            revision,
                            request.model_dump(mode="json"),
                            analysis.model_dump(mode="json", exclude={"elapsed_ms"}),
                        ]
                    )
                    stored = disk.get(cache_key)
                    if stored:
                        evidence = [EvidenceResult.model_validate(row) for row in stored]
                        reused = True
                        stages["selection_method"] = "persistent_cache_revalidated"
                if (
                    not reused
                    and memory
                    and hasattr(retrieval, "by_ids")
                    and not request.filters.model_dump(exclude_none=True)
                ):
                    recent = memory[1].search_reviewed(
                        memory[0], request.query, None, memory[2], get_embedding_service()
                    )
                    ids = {ref for row in recent for ref in row.metadata.get("kb_refs", [])}
                    candidates = retrieval.by_ids(ids) if ids else []
                    # An old repair is never a diagnosis: retain only currently applicable sources.
                    evidence = [
                        row
                        for row in candidates
                        if (procedure := parse_procedure(row))
                        and applicability_issue(procedure, analysis) is None
                    ]
                    if evidence:
                        history = recent
                        reused = True
                        stages["selection_method"] = "reviewed_history_revalidated"
                if not reused and memory and analysis.category:
                    signature = self.reuse.signature(analysis, request, memory[2], self.settings)
                    if signature:
                        vector = get_embedding_service().embed_query(search.query)
                        cached = self.reuse.get(
                            signature, vector, self.settings.semantic_reuse_threshold
                        )
                        if cached is not None:
                            evidence, reused = cached, True
                if not reused:
                    evidence = retrieval.search(search).results
            except EmbeddingInputTooLong:
                if search.rerank:
                    # Optional pairwise reranking has a smaller query budget.
                    # Pooled retrieval still considers every complaint segment.
                    search = search.model_copy(update={"rerank": False, "rerank_k": None})
                    rerank_disabled = True
                elif search.query != request.query:
                    search = search.model_copy(update={"query": request.query})
                else:
                    raise
                evidence = retrieval.search(search).results
            stages["retrieved_kb_ids"] = list(dict.fromkeys(row.doc_id for row in evidence))
            stages["eligibility"] = [
                {
                    "doc_id": row.doc_id,
                    "rejection": applicability_issue(procedure, analysis)
                    if (procedure := parse_procedure(row))
                    else "invalid_procedure",
                }
                for row in evidence
            ]
            if evidence:
                history_search = search.model_copy(
                    update={
                        "filters": filters.model_copy(update={"doc_type": DocType.RESOLVED_TICKET}),
                        "rerank": False,
                    }
                )
                history = history or retrieval.search(history_search).results
                evidence = rank_fallback(evidence, analysis)
                if self.settings.llm_enabled and self.settings.llm_selection_enabled and not reused:
                    try:
                        evidence = select_relevant_procedure(
                            evidence,
                            request.query,
                            analysis,
                            self.language,
                            retain_alternatives=True,
                        )
                        stages["selection_method"] = "language_selection"
                    except LanguageUnavailable as exc:
                        selection_error = str(exc)
        stages["selection_error"] = selection_error
        stages["selected_pool_ids"] = list(dict.fromkeys(row.doc_id for row in evidence))
        result = draft_resolution(analysis, evidence, max_sources=request.max_sources)
        result = finalize_resolution(result, evidence, analysis)
        stages["rerank_disabled_for_long_query"] = rerank_disabled
        if rerank_disabled:
            result.limitations.append(
                "Optional cross-encoder reranking skipped for the long complaint; "
                "hybrid retrieval used pooled embeddings of all query segments."
            )
        if reused:
            result.evidence_reuse = {
                "persistent_cache_revalidated": "exact_cache_revalidated",
                "reviewed_history_revalidated": "reviewed_history_revalidated",
            }.get(stages["selection_method"], "semantic_hit_revalidated")
        if selection_error:
            result.limitations.append(
                "Procedure selection unavailable; local ranking retained: " + selection_error
            )
        if result.sources:
            if memory:
                history = (
                    memory[1].search_reviewed(
                        memory[0],
                        request.query,
                        [s.doc_id for s in result.sources],
                        memory[2],
                        get_embedding_service(),
                    )
                    + history
                )
            cited_ids = {source.doc_id for source in result.sources}
            categories = {
                row.metadata.get("category") for row in evidence if row.doc_id in cited_ids
            }
            result.historical_cases = select_history(
                history, result.sources, related_categories=categories - {None}
            )
            result.customer_plan = customer_plan(result)
            result = finalize_resolution(result, evidence, analysis, historical_evidence=history)
        if result.validation.status == "passed" and result.sources and signature:
            self.reuse.put(signature, vector, evidence)
        if cache_key and result.validation.status == "passed" and result.sources:
            # Persist only public KB evidence, never private conversation text or inferred diagnoses.
            disk.put(cache_key, [row.model_dump(mode="json") for row in evidence])
        result = add_language_draft(result, request.query, self.settings, self.language)
        stages["final_source_ids"] = [source.doc_id for source in result.sources]
        stages["historical_case_ids"] = [row.doc_id for row in result.historical_cases]
        logger.info("Resolution stage trace %s", json.dumps(stages, sort_keys=True))
        logger.info(
            "Resolution stages retrieved=%d eligible=%d selected=%d final=%d method=%s",
            len(stages["retrieved_kb_ids"]),
            sum(row["rejection"] is None for row in stages["eligibility"]),
            len(stages["selected_pool_ids"]),
            len(stages["final_source_ids"]),
            stages["selection_method"],
        )
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
