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
    known = {(fact.name, fact.value) for fact in analysis.reported_facts}
    finding = procedure.quotes["condition"].text
    if ("port_status", "rejected") in known and re.search(
        r"\b(?:order is accepted|no failure is recorded)\b", finding, re.I
    ):
        return "accepted_port_gate_for_rejected_transfer"
    if (
        ("optical_signal", "normal_reported") in known
        and re.search(
            r"\b(?:no received signal|no optical signal|no light received)\b", finding, re.I
        )
    ):
        return "total_optical_loss_gate_for_intermittent_symptom"
    if ("tv_symptom", "buffering") in known and re.search(
        r"\b(?:entitlement|purchase|billing|display cable|local menu)\b", finding, re.I
    ):
        return "channel_entitlement_procedure_for_buffering"
    if ("sms_scope", "one_sender") in known and re.search(
        r"\b(?:service.center|donor|porting)\b", finding, re.I
    ):
        return "general_sms_procedure_for_single_sender_failure"
    if ("billing_status", "pending") in known and re.search(
        r"\b(?:settled|ledger is paid|ledger confirms duplicate)\b", finding, re.I
    ):
        return "settled_payment_procedure_for_pending_payment"
    if ("wired_connection", "failing") in known and re.search(
        r"\b(?:wired|Ethernet|WAN)\b[^;.]{0,35}\b(?:normal|stable|working)\b", finding, re.I
    ):
        return "working_wired_gate_conflicts_with_reported_failure"
    if ("wireless_devices", "all") in known and re.search(
        r"\b(?:appliance|other clients authenticate|other devices work)\b", finding, re.I
    ):
        return "single_device_gate_for_all_devices_failure"
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
