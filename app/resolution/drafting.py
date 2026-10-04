"""Compose conditional agent drafts with citations and prior-action awareness."""

from app.resolution.applicability import applicability_issue
from app.resolution.customer import customer_plan
from app.resolution.evidence import parse_procedure, supported_scopes
from app.resolution.models import ConditionalSuggestion, DraftSource, ResolutionResponse
from app.resolution.policy import choose_decision, questions_for
from app.resolution.rendering import render_draft
from app.resolution.selection import action_key
from app.understanding.signals import extract_actions


def draft_resolution(analysis, evidence, *, max_sources=3, elapsed_ms=0):
    """Copy supported procedure fields; never infer that a diagnostic gate has been met."""
    attempted = {item.action for item in analysis.actions if item.status == "attempted"}
    acknowledged = list(
        dict.fromkeys(item.text for item in analysis.actions if item.status == "attempted")
    )
    scopes = supported_scopes(analysis)
    sources = []
    suggestions = []
    seen = set()
    remedies = set()
    # Category is an uncertain label, not a diagnostic finding. Preserve the
    # relevance ranking and enforce observed service/condition compatibility.
    for result in evidence:
        procedure = parse_procedure(result)
        if (
            procedure is None
            or applicability_issue(procedure, analysis) is not None
            or result.doc_id in seen
            or action_key(procedure) in remedies
        ):
            continue
        seen.add(result.doc_id)
        remedies.add(action_key(procedure))
        citation_id = f"S{len(sources) + 1}"
        action = procedure.quotes["action"].text
        repeated = sorted(attempted & {item.action for item in extract_actions(action)})
        suggestions.append(
            ConditionalSuggestion(
                citation_id=citation_id,
                required_finding=procedure.quotes["condition"].text,
                proposed_action=action,
                restriction=procedure.quotes["restriction"].text,
                status="withheld_previously_attempted"
                if repeated
                else "requires_agent_confirmation",
                repeated_actions=repeated,
            )
        )
        sources.append(
            DraftSource(
                citation_id=citation_id,
                doc_id=result.doc_id,
                chunk_id=result.chunk_id,
                title=result.title,
                is_synthetic=True,
                authority=result.metadata["authority"],
                quotes=list(procedure.quotes.values()),
                quote_scope="parent_document" if result.evidence_content else "chunk",
            )
        )
        if len(sources) == max_sources:
            break
    questions = questions_for(analysis)
    contact = any(item.kind == "contact_support" for item in analysis.requests)
    result = ResolutionResponse(
        status="unsupported_request"
        if analysis.scope_status == "unsupported"
        else "needs_diagnostic_confirmation"
        if suggestions
        else ("needs_clarification" if not scopes else "insufficient_evidence"),
        draft="",
        analysis=analysis,
        suggestions=suggestions,
        sources=sources,
        clarification_questions=questions,
        acknowledged_actions=acknowledged,
        contact_status="unverified" if contact else "not_requested",
        limitations=[
            "Extractive local drafting; no generative LLM is used.",
            "Only complete authored fictional-provider KB procedures are supported by this draft version.",
            "Exact-source validation does not confirm applicability or real-world diagnostic correctness.",
        ],
        elapsed_ms=elapsed_ms,
    )
    result.decision = choose_decision(result)
    result.draft = render_draft(result)
    result.customer_plan = customer_plan(result)
    return result
