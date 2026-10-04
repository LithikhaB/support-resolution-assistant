"""Bounded resolution requests and transparent extractive draft contracts."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.resolution.history import HistoricalCase
from app.retrieval.models import RetrievalFilters
from app.understanding.models import AnalysisResponse, AnalyzeRequest, TextEvidence


class ResolutionRequest(AnalyzeRequest):
    """Request a local agent draft with optional explicit metadata constraints."""

    max_sources: int = Field(default=3, ge=1, le=5, strict=True)
    rerank: bool = Field(default=True, strict=True)
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)

    @model_validator(mode="after")
    def validate_evidence_type(self):
        """Require KB procedures for drafting without silently overriding caller filters."""
        if self.filters.doc_type not in (None, "knowledge_base"):
            raise ValueError("resolution drafting requires knowledge_base evidence")
        return self


class SourceQuote(TextEvidence):
    """Trace copied fields to exact offsets in the declared chunk or parent document."""

    field: Literal[
        "scope",
        "condition",
        "action",
        "restriction",
        "verify",
        "customer_checks",
        "agent_checks",
        "escalate_if",
        "completion",
    ]


class DraftSource(BaseModel):
    """Expose the source and its synthetic provenance for every proposed procedure."""

    citation_id: str
    doc_id: str
    chunk_id: int
    title: str
    is_synthetic: bool
    authority: str
    quotes: list[SourceQuote]
    quote_scope: Literal["chunk", "parent_document"] = "chunk"


class ConditionalSuggestion(BaseModel):
    """Keep a possible action conditional on an unconfirmed diagnostic finding."""

    citation_id: str
    required_finding: str
    proposed_action: str
    restriction: str
    status: Literal["requires_agent_confirmation", "withheld_previously_attempted"]
    repeated_actions: list[str] = Field(default_factory=list)


class SupportDecision(BaseModel):
    """Recommend the next workflow action without creating an external handoff."""

    action: Literal["clarify", "agent_review", "escalate"] = "agent_review"
    priority: Literal["normal", "high", "urgent"] = "normal"
    reasons: list[str] = Field(default_factory=list)
    target: Literal["support_agent", "network_operations", "field_service"] = "support_agent"
    handoff_created: Literal[False] = False


class CitationValidation(BaseModel):
    """Report exact-source and conditional-action checks, not diagnostic correctness."""

    status: Literal["not_run", "passed", "failed"] = "not_run"
    checked_sources: int = 0
    issues: list[str] = Field(default_factory=list)
    scope: str = "Exact source spans, claim fields, conditional actions, reported-area scope checks and deterministic rendering; not real-world diagnosis or arbitrary-text entailment."


class CustomerPlan(BaseModel):
    """Keep customer guidance separate from the internal conditional agent draft."""

    title: str
    summary: str
    steps: list[str] = Field(default_factory=list)
    note: str = ""


class ResolutionResponse(BaseModel):
    """Return an inspectable local draft rather than claiming a verified repair."""

    status: Literal[
        "needs_diagnostic_confirmation",
        "needs_clarification",
        "insufficient_evidence",
        "unsupported_request",
    ]
    decision: SupportDecision = Field(default_factory=SupportDecision)
    validation: CitationValidation = Field(default_factory=CitationValidation)
    method: Literal["local_extractive_v1"] = "local_extractive_v1"
    language_summary: str | None = None
    language_plan: CustomerPlan | None = None
    language_draft: str | None = None
    language_status: Literal["disabled", "generated_for_review", "fallback"] = "disabled"
    language_model: str | None = None
    language_provider: str | None = None
    faithfulness_status: Literal["not_run", "model_checked", "rejected"] = "not_run"
    faithfulness_issues: list[str] = Field(default_factory=list, max_length=10)
    language_error: str | None = None
    historical_cases: list[HistoricalCase] = Field(default_factory=list)
    agent_review_required: Literal[True] = True
    draft: str
    analysis: AnalysisResponse
    suggestions: list[ConditionalSuggestion]
    sources: list[DraftSource]
    clarification_questions: list[str]
    acknowledged_actions: list[str]
    contact_status: Literal["not_requested", "unverified"]
    customer_plan: CustomerPlan | None = None
    limitations: list[str]
    elapsed_ms: float = Field(ge=0, allow_inf_nan=False)
