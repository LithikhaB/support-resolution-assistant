"""Render reviewed structured fields without accepting arbitrary generated prose."""

import re

from app.resolution.customer import physical_damage


def customer_text(text):
    """Keep customer-supplied citation-like markers visually distinct from source aliases."""
    return re.sub(r"\[S(\d+)\]", r"(customer text: S\1)", text)


def render_draft(response):
    """Render the current local contract while retaining conditional language and restrictions."""
    paragraphs = ["Draft for agent review. A cause has not been confirmed."]
    if response.analysis.scope_status == "unsupported":
        paragraphs.append(
            "This request is outside the supported telecom complaint workflow. No repair is recommended; a support agent can review the request."
        )
    if physical_damage(response.analysis) and response.analysis.scope_status != "unsupported":
        paragraphs.append(
            "Reported physical damage requires technician inspection and a replacement assessment. Routine connection procedures are withheld. Area recovery does not establish equipment safety; no replacement has been booked."
        )
    if response.acknowledged_actions:
        paragraphs.append(
            "Already attempted, according to the customer: "
            + "; ".join(customer_text(text) for text in response.acknowledged_actions)
        )
    if response.analysis.reported_facts:
        paragraphs.append(
            "Customer-reported observations: "
            + "; ".join(customer_text(item.text) for item in response.analysis.reported_facts)
        )
    for question in response.clarification_questions:
        paragraphs.append("Clarify: " + question)
    visible_suggestions = response.suggestions
    if any(f.name == "impact" and f.value == "working" for f in response.analysis.reported_facts):
        paragraphs.append(
            "The customer reports recovery. Ask an agent to review the outcome before closing the issue; no further repair is recommended."
        )
        visible_suggestions = []
    if response.decision.action == "clarify" and response.suggestions:
        paragraphs.append(
            "Answer the clarification questions before choosing a repair. Retrieved procedures are attached for agent review, not recommended actions yet."
        )
        visible_suggestions = []
    for item in visible_suggestions:
        if item.repeated_actions:
            paragraphs.append(
                f"[{item.citation_id}] A retrieved procedure includes an action already reported as attempted. Withhold that repeat and ask the agent to review the earlier result. Required finding: {item.required_finding}"
            )
        else:
            paragraphs.append(
                f"[{item.citation_id}] Ask an authorized agent to verify: {item.required_finding} Only if confirmed, the procedure proposes: {item.proposed_action}"
            )
        paragraphs.append(f"[{item.citation_id}] Restriction: {item.restriction}")
    if (
        not response.suggestions
        and response.analysis.scope_status != "unsupported"
        and not physical_damage(response.analysis)
    ):
        paragraphs.append(
            "No complete applicable procedure was selected. Obtain the missing service details or route to an agent for investigation."
        )
    if response.contact_status == "unverified":
        paragraphs.append(
            "The customer requested human support. Arrange a follow-up through the support desk; this demonstration has no telephone directory."
        )
    if response.suggestions:
        paragraphs.append(
            "The cited procedures are fictional-provider examples and require review before use with a real provider."
        )
    if response.validation.status == "failed":
        paragraphs.append(
            "Citation validation failed. Proposed procedures were withheld; a support agent must review the evidence."
        )
    decision = response.decision
    paragraphs.append(
        f"Next action: {decision.action}. Priority: {decision.priority}. Recommended reviewer: {decision.target}. No external handoff has been created."
    )
    return "\n\n".join(paragraphs)
