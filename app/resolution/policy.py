"""Explain when to clarify, request agent review or recommend escalation."""

from app.resolution.customer import physical_damage
from app.resolution.evidence import supported_scopes
from app.resolution.models import SupportDecision
from app.understanding.context import AREA_OUTAGE_QUESTION


def choose_decision(response):
    """Base urgency on stated impact and keep escalation advisory until integrations exist."""
    severity = response.analysis.severity.value
    priority = {"critical": "urgent", "high": "high"}.get(severity, "normal")
    if any(
        f.name == "security_request" and f.value == "otp_sharing"
        for f in response.analysis.reported_facts
    ):
        return SupportDecision(
            action="escalate", priority="high", reasons=["verification_code_sharing_request"]
        )
    if response.validation.status == "failed":
        return SupportDecision(
            action="escalate", priority=priority, reasons=["citation_validation_failed"]
        )
    if response.analysis.scope_status == "unsupported":
        return SupportDecision(action="agent_review", reasons=["unsupported_request"])
    if {p.product for p in response.analysis.products} == {"landline"}:
        return SupportDecision(
            action="escalate", priority=priority, reasons=["landline_knowledge_gap"]
        )
    if physical_damage(response.analysis):
        return SupportDecision(
            action="escalate",
            priority="urgent" if severity == "critical" else "high",
            target="field_service",
            reasons=[
                "reported_physical_equipment_damage",
                "inspection_and_replacement_assessment_required",
            ],
        )
    if severity == "critical":
        return SupportDecision(
            action="escalate",
            priority="urgent",
            target="network_operations",
            reasons=["reported_area_outage"],
        )
    if response.contact_status == "unverified":
        return SupportDecision(
            action="escalate",
            priority=priority,
            reasons=["customer_requested_support_contact"],
        )
    if response.status == "insufficient_evidence":
        return SupportDecision(
            action="escalate", priority=priority, reasons=["no_applicable_complete_procedure"]
        )
    if response.suggestions and all(
        s.status == "withheld_previously_attempted" for s in response.suggestions
    ):
        return SupportDecision(
            action="escalate", priority=priority, reasons=["available_actions_already_attempted"]
        )
    if response.clarification_questions:
        return SupportDecision(
            action="clarify", priority=priority, reasons=["missing_customer_observations"]
        )
    return SupportDecision(
        action="agent_review", priority=priority, reasons=["diagnostic_confirmation_required"]
    )


def questions_for(analysis):
    """Retain missing observations and explicitly surface contradictory reported facts."""
    questions = list(analysis.clarification_questions)
    if any(
        f.name == "security_request" and f.value == "otp_sharing" for f in analysis.reported_facts
    ):
        return []
    if analysis.scope_status == "unsupported":
        return []
    if any(request.kind == "contact_support" for request in analysis.requests):
        return []
    if {p.product for p in analysis.products} == {"landline"}:
        return []
    if analysis.category == "number_porting":
        known = {(f.name, f.value) for f in analysis.reported_facts}
        if ("port_status", "rejected") in known:
            return [] if ("port_reason", "account_mismatch") in known else [
                "What rejection reason did the provider give? Do not share account identifiers or authorization codes."
            ]
        return ["Is the number transfer pending or rejected, and what status has the provider given?"]
    if analysis.severity.rule == "reported_area_outage":
        known = {f.name for f in analysis.reported_facts}
        questions = (
            []
            if {"area", "started"} <= known
            else [
                "Which area is affected? Do not delay incident review while collecting these details."
                if "started" in known
                else AREA_OUTAGE_QUESTION
            ]
        )
    if not supported_scopes(analysis):
        questions = (
            []
            if any(r.kind == "contact_support" for r in analysis.requests)
            else ["Which service is affected, and what exactly happens when you use it?"]
        )
    values = {}
    for fact in analysis.reported_facts:
        values.setdefault(fact.name, set()).add(fact.value)
    for name, reported in sorted(values.items()):
        if name == "equipment_condition" and reported <= {"damaged", "water_exposed"}:
            continue
        if len(reported) > 1:
            questions.append(
                f"You reported conflicting {name.replace('_', ' ')} observations. Which observation is current?"
            )
    if physical_damage(analysis):
        questions = [q for q in questions if q.startswith("You reported conflicting")]
    return list(dict.fromkeys(questions))
