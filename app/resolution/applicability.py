"""Reject procedures that conflict with the explicitly reported incident scope."""

import re

from app.resolution.customer import physical_damage
from app.resolution.evidence import supported_scopes
from app.understanding.signals import negated


def needs_baseline(analysis):
    known = {(f.name, f.value) for f in analysis.reported_facts}
    return (
        analysis.category_basis == "explicit_report"
        and analysis.category in {"broadband_outage", "slow_broadband"}
        and not any(
            f.name in {"account_change", "optical_signal", "weather_context", "wired_connection"}
            for f in analysis.reported_facts
        )
        and not (analysis.category == "slow_broadband" and ("timing", "peak_hours") in known)
        and analysis.severity.rule != "reported_area_outage"
    )


def applicability_issue(procedure, analysis):
    """Apply narrow contextual exclusions without claiming general diagnostic entailment."""
    if analysis.scope_status == "unsupported":
        return "unsupported_request"
    if physical_damage(analysis):
        return "physical_damage_requires_inspection"
    baseline_scope = (
        procedure.baseline_for is not None
        and procedure.scope == "broadband"
        and bool(supported_scopes(analysis) & {"fibre_broadband", "home_wifi", "router"})
    )
    if procedure.scope not in supported_scopes(analysis) and not baseline_scope:
        return "different_service_scope"
    known = {(fact.name, fact.value) for fact in analysis.reported_facts}
    finding = procedure.quotes["condition"].text
    if (
        re.search(r"\b(?:overheats?|overheating|hot|heat)\b", finding, re.I)
        and ("equipment_condition", "overheating") not in known
    ):
        return "overheating_gate_without_reported_heat_context"
    if ("charge", "late_fee") in known and not re.search(
        r"\b(?:late fee|due date|late payment)\b", finding, re.I
    ):
        return "unrelated_procedure_for_reported_late_fee"
    if (
        analysis.category == "billing_dispute"
        and analysis.severity.rule not in {"reported_complete_loss", "reported_area_outage"}
        and re.search(
            r"\b(?:restoration|provisioning|service is restored|restriction|reallocation)\b",
            finding + " " + procedure.quotes["action"].text,
            re.I,
        )
    ):
        return "restoration_procedure_without_reported_service_loss"
    if (
        analysis.category in {"intermittent_broadband", "slow_broadband", "wifi_connectivity"}
        and ("weather_context", "reported") not in known
        and re.search(r"\b(?:moisture|rain|wet.weather)\b", finding, re.I)
    ):
        return "weather_gate_without_reported_weather_context"
    if (
        ("connection_pattern", "intermittent") in known
        and analysis.category in {"wifi_connectivity", "intermittent_broadband", None}
        and re.search(
            r"\b(?:saved wireless profile|other clients authenticate|band the appliance does not support)\b",
            finding,
            re.I,
        )
    ):
        return "authentication_gate_for_reported_wireless_drops"
    if procedure.baseline_for and procedure.baseline_for != analysis.category:
        return "different_baseline_category"
    if needs_baseline(analysis) and procedure.baseline_for != analysis.category:
        return "specific_procedure_without_reported_context_use_baseline"
    if (
        ("timing", "persistent_daytime") in known
        and ("timing", "peak_hours") not in known
        and re.search(r"\b(?:peak|evening|congestion)\b", finding, re.I)
    ):
        return "peak_timing_gate_for_persistent_daytime_slowness"
    if {
        ("payment_scope", "duplicate_same_invoice"),
        ("billing_status", "settled"),
    } <= known and not re.search(
        r"\b(?:duplicate settled|two settled)\s+payments\b", finding, re.I
    ):
        return "different_payment_issue_for_reported_duplicate_settlement"
    if ("optical_signal", "loss_reported") in known and not re.search(
        r"\b(?:no received signal|no optical signal|no light received)\b", finding, re.I
    ):
        return "defer_other_procedures_until_reported_optical_alarm_checked"
    if ("billing_status", "settled") in known and re.search(
        r"\b(?:authorization only|no settlement confirmation)\b", finding, re.I
    ):
        return "pending_payment_gate_for_reported_settlement"
    if (
        ("weather_context", "reported") in known
        and ("timing", "peak_hours") not in known
        and re.search(r"\b(?:peak|evening|capacity|congestion)\b", finding, re.I)
    ):
        return "peak_hour_gate_without_peak_timing_in_weather_report"
    if ("port_status", "rejected") in known and re.search(
        r"\b(?:order is accepted|no failure is recorded)\b", finding, re.I
    ):
        return "accepted_port_gate_for_rejected_transfer"
    if ("optical_signal", "normal_reported") in known and re.search(
        r"\b(?:no received signal|no optical signal|no light received)\b", finding, re.I
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
    if needs_baseline(analysis):
        baseline = (
            "Broadband unavailable: establish the fault before choosing a repair"
            if analysis.category == "broadband_outage"
            else "Persistent broadband slowness: establish a controlled baseline"
        )
        return query + " " + baseline
    if supported_scopes(analysis) == {"home_wifi"}:
        return query + " Ethernet is working; investigate wireless Wi-Fi connectivity only."
    if analysis.severity.rule == "reported_area_outage":
        return (
            query + " Regional outage affecting multiple buildings; check provider major incident."
        )
    if any(f.name == "weather_context" for f in analysis.reported_facts):
        return (
            query
            + " Weather-related intermittent service: investigate line quality and physical faults; cause unconfirmed."
        )
    return query
