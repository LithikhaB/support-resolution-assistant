"""Reject procedures that conflict with the explicitly reported incident scope."""

import re

from app.resolution.customer import physical_damage
from app.resolution.evidence import supported_scopes
from app.understanding.signals import negated


def applicability_issue(procedure, analysis):
    """Apply narrow contextual exclusions without claiming general diagnostic entailment."""
    if analysis.scope_status == "unsupported":
        return "unsupported_request"
    if physical_damage(analysis):
        return "physical_damage_requires_inspection"
    if procedure.scope not in supported_scopes(analysis):
        return "different_service_scope"
    if analysis.severity.rule == "reported_area_outage":
        finding = procedure.quotes["condition"].text
        matches = re.finditer(
            r"\b(?:regional|major incident|multiple (?:buildings|sites|areas)|area.wide|network.wide)\b",
            finding,
            re.I,
        )
        if not any(not negated(finding[: match.start()]) for match in matches):
            return "local_procedure_for_reported_area_outage"
    return None


def evidence_query(query, analysis):
    """Add observed incident scope to retrieval without asserting an underlying diagnosis."""
    if supported_scopes(analysis) == {"home_wifi"}:
        return query + " Ethernet is working; investigate wireless Wi-Fi connectivity only."
    if analysis.severity.rule == "reported_area_outage":
        return (
            query + " Regional outage affecting multiple buildings; check provider major incident."
        )
    return query
