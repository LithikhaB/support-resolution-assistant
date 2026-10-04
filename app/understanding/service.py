"""Compose local category inference with independently explainable text rules."""

import logging
from functools import lru_cache
from threading import Lock
from time import perf_counter

from app.config.settings import get_settings
from app.ingestion.artifacts import digest
from app.llm.client import LanguageUnavailable
from app.llm.providers import ProviderChain, get_language_client, last_provider
from app.understanding.classifier import CategoryClassifier, UnderstandingUnavailable
from app.understanding.context import clarification_questions, extract_facts, extract_requests
from app.understanding.language import interpret_complaint
from app.understanding.models import AnalysisResponse, AnalyzeRequest, RuleAssessment, TextEvidence
from app.understanding.routing import (
    compatible_category,
    explicit_category,
    load_category_products,
    load_policy,
)
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

    def __init__(self, classifier: CategoryClassifier, *, settings=None, language=None):
        self.classifier = classifier
        self.settings = settings or get_settings()
        self.language = language or (
            ProviderChain(self.settings) if settings is not None else get_language_client()
        )
        self.model_version = digest(classifier.artifact.model_dump(mode="json"))[:16]
        self.routing = load_policy(self.settings, self.model_version)
        try:
            self.category_products = load_category_products(self.settings.category_products_path)
        except (ValueError, OSError):
            raise UnderstandingUnavailable("invalid_category_mapping") from None

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
        facts = extract_facts(request.query)
        language_method, language_error = "rules_v1", None
        language_category = None
        language_severity = language_sentiment = None
        if self.settings.llm_enabled:
            try:
                (
                    interpreted_products,
                    facts,
                    language_category,
                    language_severity,
                    language_sentiment,
                ) = interpret_complaint(
                    request.query,
                    self.language or get_language_client(),
                    category_options=sorted(self.category_products),
                    category_products=self.category_products,
                    include_assessments=True,
                )
                products = interpreted_products or products
                trace = last_provider.get() or {
                    "provider": "groq",
                    "model": self.settings.groq_model,
                }
                language_method = trace["provider"] + "_extraction_v1"
                if products and scope_status != "unsupported":
                    scope_status, scope_reason = "supported", "interpreted_service_mention"
            except LanguageUnavailable as exc:
                language_method, language_error = "rules_fallback", str(exc)
        accepted = accepted and compatible_category(
            candidates[0].category, products, self.category_products
        )
        severity = assess_severity(request.query)
        if severity.value == "unknown" and language_severity is not None:
            severity = language_severity
        sentiment = assess_sentiment(request.query)
        if sentiment.value == "unknown" and language_sentiment is not None:
            sentiment = language_sentiment
        if severity.value == "unknown":
            impacts = [fact for fact in facts if fact.name == "impact"]
            values = {fact.value for fact in impacts}
            if len(values) == 1:
                impact = impacts[-1]
                value = {
                    "complete_loss": "high",
                    "intermittent": "medium",
                    "degraded": "medium",
                    "working": "low",
                }.get(impact.value)
                if value:
                    severity = RuleAssessment(
                        value=value,
                        rule="quoted_service_impact",
                        evidence=[
                            TextEvidence(**impact.model_dump(include={"text", "start", "end"}))
                        ],
                    )
        reported_category = explicit_category(products, severity)
        intermittent = [f for f in facts if f.name == "connection_pattern"]
        wired = {f.value for f in facts if f.name == "wired_connection"}
        if (
            not reported_category
            and intermittent
            and wired != {"working"}
            and any(p.product == "broadband" for p in products)
        ):
            reported_category = "intermittent_broadband"
        if wired == {"working"} and any(p.product == "home_wifi" for p in products):
            reported_category = "wifi_connectivity"
        category = reported_category or (candidates[0].category if accepted else None)
        category_evidence = (
            [TextEvidence(**f.model_dump(include={"text", "start", "end"})) for f in intermittent]
            if reported_category in {"intermittent_broadband", "wifi_connectivity"}
            else severity.evidence
            if reported_category
            else []
        )
        category_basis = (
            "explicit_report" if reported_category else ("model" if category else "uncertain")
        )
        if language_category is not None and not reported_category:
            category = language_category.category
            category_basis = "language_assisted"
            start = request.query.rfind(language_category.quote)
            if start < 0:
                start = request.query.lower().rfind(language_category.quote.lower())
            if start >= 0:
                quote_text = request.query[start : start + len(language_category.quote)]
                category_evidence = [
                    TextEvidence(
                        text=quote_text,
                        start=start,
                        end=start + len(quote_text),
                    )
                ]
            else:
                category_evidence = [
                    TextEvidence(
                        text=request.query[: min(len(request.query), len(language_category.quote))],
                        start=0,
                        end=min(len(request.query), len(language_category.quote)),
                    )
                ]
        if scope_status == "unsupported":
            category = reported_category = None
            category_basis, category_evidence = "uncertain", []
        accepted = category is not None
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
            category_basis=category_basis,
            category_evidence=category_evidence,
            routing_policy="development_selected_v1" if self.routing else "fixed_v1",
            category_status="predicted" if accepted else "uncertain",
            candidates=candidates,
            products=products,
            severity=severity,
            sentiment=sentiment,
            actions=extract_actions(request.query),
            reported_facts=facts,
            requests=requests,
            needs_clarification=bool(questions),
            clarification_questions=questions,
            model_version=self.model_version,
            language_method=language_method,
            language_provider=trace["provider"]
            if language_method.endswith("extraction_v1")
            else None,
            language_model=trace["model"] if language_method.endswith("extraction_v1") else None,
            language_error=language_error,
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
