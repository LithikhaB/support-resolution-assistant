"""Validated complaint-analysis contracts with explicit uncertainty and provenance."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AnalyzeRequest(BaseModel):
    """Accept a bounded complaint without requiring retrieval or database access."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    query: str = Field(min_length=1, max_length=10000)


class CategoryCandidate(BaseModel):
    """Expose an uncalibrated classifier score, not a probability of correctness."""

    category: str
    score: float = Field(ge=0, le=1, allow_inf_nan=False)


class TextEvidence(BaseModel):
    """Locate a matched observation in the normalized request text."""

    text: str
    start: int = Field(ge=0)
    end: int = Field(ge=1)


class RuleAssessment(BaseModel):
    """Explain a conservative heuristic assessment through its matched text."""

    value: str
    rule: str
    evidence: list[TextEvidence] = Field(default_factory=list)
    method: Literal["rules_v1"] = "rules_v1"


class ActionObservation(TextEvidence):
    """Distinguish completed troubleshooting from negated or suggested actions."""

    action: str
    status: Literal["attempted", "not_attempted", "suggested"]


class ProductObservation(TextEvidence):
    """Identify a product mention without inferring a technical diagnosis."""

    product: str


class ReportedFact(TextEvidence):
    """Preserve a customer statement without treating it as verified diagnostics."""

    name: str
    value: str


class CustomerRequest(TextEvidence):
    """Identify an explicit support request without inventing contact information."""

    kind: Literal["contact_support", "next_steps", "replacement"]


class AnalysisResponse(BaseModel):
    """Return a provisional category and independently extracted complaint context."""

    category: str | None
    scope_status: Literal["supported", "unsupported", "uncertain"] = "uncertain"
    scope_reason: str = "not_assessed"
    category_status: Literal["predicted", "uncertain"]
    category_basis: Literal["model", "explicit_report", "uncertain"] = "uncertain"
    category_evidence: list[TextEvidence] = Field(default_factory=list)
    routing_policy: str = "fixed_v1"
    candidates: list[CategoryCandidate]
    score_note: str = (
        "Uncalibrated classifier scores; thresholds do not guarantee out-of-scope detection."
    )
    products: list[ProductObservation]
    severity: RuleAssessment
    sentiment: RuleAssessment
    actions: list[ActionObservation]
    reported_facts: list[ReportedFact] = Field(default_factory=list)
    requests: list[CustomerRequest] = Field(default_factory=list)
    needs_clarification: bool
    clarification_questions: list[str]
    model_version: str
    language_method: Literal[
        "rules_v1", "groq_extraction_v1", "gemini_extraction_v1", "rules_fallback"
    ] = "rules_v1"
    language_provider: str | None = None
    language_model: str | None = None
    language_error: str | None = None
    elapsed_ms: float = Field(ge=0, allow_inf_nan=False)
