"""Extract reported observations and choose questions from the available context."""

import re

from app.understanding.models import CustomerRequest, ReportedFact
from app.understanding.signals import negated, span

FACT_PATTERNS = (
    (
        "cable_condition",
        "intact",
        r"\bcables?(?:\s+(?:in|at)\s+(?:(?:my|the)\s+)?(?:home|house))?\s+(?:is|are|looks?|seems?)\s+(?:physically\s+)?(?:intact|undamaged|fine)\b",
    ),
    (
        "cable_condition",
        "damaged",
        r"\bcables?(?:\s+(?:in|at)\s+(?:(?:my|the)\s+)?(?:home|house))?\s+(?:is|are|looks?|seems?)\s+(?:damaged|broken|frayed)\b",
    ),
    (
        "wired_connection",
        "working",
        r"\b(?:Ethernet|wired(?: connection)?)\s+(?:is\s+)?(?:works?|working|stable|fine)\b",
    ),
    (
        "connection_pattern",
        "intermittent",
        r"\b(?:broadband|internet|connection|wi[ -]?fi)\s+(?:keeps?\s+)?(?:dropping|drops|disconnects|disconnecting|cuts out)\b",
    ),
)


def extract_facts(text: str) -> list[ReportedFact]:
    """Keep explicit positive statements, excluding questions, conditions and negations."""
    facts = []
    for clause in re.finditer(r"[^.!?;,]+[.!?;,]?", text):
        fragment = clause.group()
        if fragment.rstrip().endswith("?") or re.search(
            r"\b(?:if|whether|suppose|might|maybe)\b", fragment, re.I
        ):
            continue
        for name, value, pattern in FACT_PATTERNS:
            for match in re.finditer(pattern, fragment, re.I):
                if negated(fragment[: match.start()]):
                    continue
                facts.append(
                    ReportedFact(
                        **span(text, match, clause.start()).model_dump(), name=name, value=value
                    )
                )
    return sorted(facts, key=lambda item: (item.start, item.name))


def extract_requests(text: str) -> list[CustomerRequest]:
    """Recognize requests for contact details and further help using explicit wording."""
    requests = []
    patterns = {
        "contact_support": r"\b(?:helpline(?: number)?|support (?:phone |contact )?number|contact (?:support|an agent|a human)|speak to (?:an agent|a human))\b",
        "next_steps": r"\b(?:what (?:should I do|to do)(?: now| next)?|next steps?|what can I (?:do|try))\b",
    }
    for kind, pattern in patterns.items():
        for match in re.finditer(pattern, text, re.I):
            if not negated(text[max(0, match.start() - 45) : match.start()]):
                requests.append(CustomerRequest(**span(text, match).model_dump(), kind=kind))
    return sorted(requests, key=lambda item: item.start)


def clarification_questions(*, accepted, products, severity, facts, requests) -> list[str]:
    """Ask for missing diagnostic context rather than repeating known service details."""
    questions = []
    services = {item.product for item in products}
    known = {(item.name, item.value) for item in facts}
    if not accepted:
        if services & {"broadband", "home_wifi", "router"}:
            if ("wired_connection", "working") in known:
                questions.append(
                    "Does the Wi-Fi problem affect one device or every wireless device?"
                )
            else:
                questions.append(
                    "During a drop, does a device connected by Ethernet also lose internet, or is only Wi-Fi affected?"
                )
        elif "billing" in services:
            questions.append("Which charge or payment is affected, and is it pending or settled?")
        elif "mobile" in services:
            questions.append("Are calls, texts, mobile data, or several of these affected?")
        elif "iptv" in services:
            questions.append("Is the TV showing no picture, an error, or buffering?")
        else:
            questions.append("Which service is affected, and what exactly happens when you use it?")
    if severity.value == "unknown":
        questions.append(
            "Is service completely unavailable or intermittent, and which devices or people are affected?"
        )
    if any(item.kind == "contact_support" for item in requests):
        questions.append("Which provider and country or region should the support contact be for?")
    return questions
