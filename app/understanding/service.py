"""Compose local category inference with independently explainable text rules."""

import logging
from functools import lru_cache
from threading import Lock
from time import perf_counter

from app.config.settings import get_settings
from app.ingestion.artifacts import digest
from app.understanding.classifier import CategoryClassifier
from app.understanding.context import clarification_questions, extract_facts, extract_requests
from app.understanding.models import AnalysisResponse, AnalyzeRequest
from app.understanding.routing import compatible_category, explicit_category, load_policy
from app.understanding.scope import scope_assessment
from app.understanding.signals import (
    assess_sentiment,
    assess_severity,
    extract_actions,
    extract_products,
)

logger = logging.getLogger(__name__)
_factory_lock = Lock()


class UnderstandingService:
    """Analyze complaints without treating predicted categories as confirmed diagnoses."""

    def __init__(self, classifier: CategoryClassifier, *, settings=None):
        self.classifier = classifier
        self.settings = settings or get_settings()
        self.model_version = digest(classifier.artifact.model_dump(mode="json"))[:16]
        self.routing = load_policy(self.settings, self.model_version)

    def analyze(self, request: AnalyzeRequest) -> AnalysisResponse:
        """Keep category uncertainty, observed products, impact and prior actions distinct."""
        started = perf_counter()
        scope_status, scope_reason = scope_assessment(request.query)
        candidates = self.classifier.predict(request.query)
        accepted = candidates[0].score >= (
            self.routing.min_score if self.routing else self.settings.understanding_min_score
        ) and candidates[0].score - candidates[1].score >= (
            self.routing.min_margin if self.routing else self.settings.understanding_min_margin
        )
        products = extract_products(request.query)
        accepted = accepted and compatible_category(candidates[0].category, products)
        severity = assess_severity(request.query)
        reported_category = explicit_category(products, severity)
        category = reported_category or (candidates[0].category if accepted else None)
        if scope_status == "unsupported":
            category = reported_category = None
        accepted = category is not None
        facts = extract_facts(request.query)
        requests = extract_requests(request.query)
        questions = clarification_questions(
            accepted=accepted, products=products, severity=severity, facts=facts, requests=requests
        )
        if scope_status == "unsupported":
            questions = []
        result = AnalysisResponse(
            scope_status=scope_status,
            scope_reason=scope_reason,
            category=category,
            category_basis="explicit_report"
            if reported_category
            else ("model" if accepted else "uncertain"),
            category_evidence=severity.evidence if reported_category else [],
            routing_policy="development_selected_v1" if self.routing else "fixed_v1",
            category_status="predicted" if accepted else "uncertain",
            candidates=candidates,
            products=products,
            severity=severity,
            sentiment=assess_sentiment(request.query),
            actions=extract_actions(request.query),
            reported_facts=facts,
            requests=requests,
            needs_clarification=bool(questions),
            clarification_questions=questions,
            model_version=self.model_version,
            elapsed_ms=(perf_counter() - started) * 1000,
        )
        logger.info(
            "Understanding query_chars=%d category_status=%s actions=%d elapsed_ms=%.1f",
            len(request.query),
            result.category_status,
            len(result.actions),
            result.elapsed_ms,
        )
        return result


@lru_cache(maxsize=1)
def _cached_service() -> UnderstandingService:
    settings = get_settings()
    classifier = CategoryClassifier.load(settings.understanding_model_path, settings=settings)
    return UnderstandingService(classifier, settings=settings)


def get_understanding_service() -> UnderstandingService:
    """Initialize a single classifier lazily; restart the API after publishing new weights."""
    with _factory_lock:
        return _cached_service()
