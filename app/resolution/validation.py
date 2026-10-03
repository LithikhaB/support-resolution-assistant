"""Validate extractive support claims against the evidence used for this request."""

from collections import Counter

from app.resolution.applicability import applicability_issue
from app.resolution.customer import customer_plan
from app.resolution.evidence import parse_procedure, supported_scopes
from app.resolution.models import CitationValidation
from app.resolution.policy import choose_decision, questions_for
from app.resolution.rendering import render_draft
from app.understanding.signals import extract_actions


def validate_citations(response, evidence, analysis):
    """Check provenance, exact field support, action conditions and rendered output integrity."""
    issues = []
    records = {(r.doc_id, r.chunk_id): r for r in evidence}
    if len(records) != len(evidence):
        issues.append("ambiguous_retrieved_source")
    aliases = [s.citation_id for s in response.sources]
    if len(set(aliases)) != len(aliases):
        issues.append("duplicate_citation_alias")
    if aliases != [f"S{i}" for i in range(1, len(aliases) + 1)]:
        issues.append("invalid_citation_aliases")
    attempted = {a.action for a in analysis.actions if a.status == "attempted"}
    expected_actions = list(
        dict.fromkeys(a.text for a in analysis.actions if a.status == "attempted")
    )
    if response.analysis != analysis or response.acknowledged_actions != expected_actions:
        issues.append("customer_context_changed")
    expected_contact = (
        "unverified"
        if any(r.kind == "contact_support" for r in analysis.requests)
        else "not_requested"
    )
    if response.clarification_questions != questions_for(analysis):
        issues.append("unsupported_clarification")
    expected_state = (
        "unsupported_request"
        if analysis.scope_status == "unsupported"
        else "needs_diagnostic_confirmation"
        if response.suggestions
        else ("needs_clarification" if not supported_scopes(analysis) else "insufficient_evidence")
    )
    if response.status != expected_state:
        issues.append("unsupported_resolution_status")
    if response.contact_status != expected_contact:
        issues.append("unsupported_contact_status")
    if response.customer_plan != customer_plan(response):
        issues.append("customer_plan_changed")
    procedures = {}
    for source in response.sources:
        record = records.get((source.doc_id, source.chunk_id))
        if record is None:
            issues.append("citation_not_retrieved")
            continue
        procedure = parse_procedure(record)
        if procedure is None or applicability_issue(procedure, analysis) is not None:
            issues.append("unsupported_source_or_scope")
            continue
        if (
            source.title != record.title
            or source.authority != record.metadata.get("authority")
            or source.is_synthetic is not record.metadata.get("is_synthetic")
        ):
            issues.append("source_provenance_changed")
        if Counter(q.field for q in source.quotes) != Counter(procedure.quotes.keys()):
            issues.append("missing_or_duplicate_source_fields")
        for quote in source.quotes:
            expected = procedure.quotes.get(quote.field)
            if expected != quote or record.content[quote.start : quote.end] != quote.text:
                issues.append("source_span_mismatch")
        procedures[source.citation_id] = procedure
    if Counter(s.citation_id for s in response.suggestions) != Counter(aliases):
        issues.append("orphan_or_missing_citation")
    for suggestion in response.suggestions:
        procedure = procedures.get(suggestion.citation_id)
        if procedure is None:
            issues.append("suggestion_without_valid_source")
            continue
        for field, quote in (
            ("required_finding", "condition"),
            ("proposed_action", "action"),
            ("restriction", "restriction"),
        ):
            if getattr(suggestion, field) != procedure.quotes[quote].text:
                issues.append("unsupported_" + field)
        repeated = sorted(
            attempted & {a.action for a in extract_actions(procedure.quotes["action"].text)}
        )
        expected_status = (
            "withheld_previously_attempted" if repeated else "requires_agent_confirmation"
        )
        if suggestion.status != expected_status or suggestion.repeated_actions != repeated:
            issues.append("unsafe_action_state")
    if response.decision != choose_decision(response):
        issues.append("workflow_decision_changed")
    if response.draft != render_draft(response):
        issues.append("unstructured_draft_text")
    return CitationValidation(
        status="failed" if issues else "passed",
        checked_sources=len(response.sources),
        issues=sorted(set(issues)),
    )


def finalize_resolution(response, evidence, analysis):
    """Withhold all proposed actions when any citation or action-support check fails."""
    from app.resolution.drafting import draft_resolution

    validation = validate_citations(response, evidence, analysis)
    if validation.status == "failed":
        response = draft_resolution(analysis, [])
    response.validation = validation
    response.decision = choose_decision(response)
    response.draft = render_draft(response)
    response.customer_plan = customer_plan(response)
    return response
