"""Read complete conditional procedures from the authored fictional-provider KB format."""

import re
from dataclasses import dataclass

from app.ingestion.schema import DocType
from app.resolution.models import SourceQuote

PRODUCT_SCOPES = {
    "broadband": {"fibre_broadband", "home_wifi", "router"},
    "home_wifi": {"home_wifi", "fibre_broadband", "router"},
    "router": {"router", "home_wifi", "fibre_broadband"},
    "mobile": {
        "mobile",
        "mobile_voice",
        "mobile_data",
        "mobile_sms",
        "mobile_roaming",
        "sim",
        "esim",
    },
    "billing": {"billing"},
    "iptv": {"iptv"},
}


@dataclass(frozen=True)
class Procedure:
    """Carry exact copied spans and the declared service scope."""

    scope: str
    quotes: dict[str, SourceQuote]
    baseline_for: str | None = None


def parse_procedure(result):
    """Reject unsupported provenance, incomplete chunks and ambiguous repeated fields."""
    if result.doc_type != DocType.KNOWLEDGE_BASE:
        return None
    if (
        result.metadata.get("authority") != "fictional_provider_policy"
        or result.metadata.get("is_synthetic") is not True
    ):
        return None
    text = result.evidence_content or result.content
    patterns = {
        "scope": r"^Scope: ([^;\r\n]+); support category: [^\r\n]+$",
        "condition": r"^Diagnostic gate: ([^\r\n]+)$",
        "action": r"^Only if that finding is established: ([^\r\n]+)$",
        "restriction": r"^Restriction: ([^\r\n]+)$",
    }
    version = result.metadata.get("procedure_version")
    if version in {2, 3}:
        patterns.update(
            {
                "verify": r"^Verify: ([^\r\n]+)$",
                "customer_checks": r"^Customer checks: ([^\r\n]+)$",
                "agent_checks": r"^Agent checks: ([^\r\n]+)$",
                "escalate_if": r"^Escalate if: ([^\r\n]+)$",
                "completion": r"^Completion: ([^\r\n]+)$",
            }
        )
    quotes = {}
    for name, pattern in patterns.items():
        matches = list(re.finditer(pattern, text, re.M))
        if len(matches) != 1 or not matches[0].group(1).strip():
            return None
        match = matches[0]
        quotes[name] = SourceQuote(
            field=name, text=match.group(1), start=match.start(1), end=match.end(1)
        )
    if version == 3:
        steps = result.metadata.get("procedure", {}).get("steps")
        if not isinstance(steps, list) or len(steps) != 5:
            return None
        phases = [
            "customer_check",
            "customer_check",
            "agent_check",
            "conditional_fix",
            "completion",
        ]
        matches = list(re.finditer(r"^Step ([1-5]): ([^\r\n]+)$", text, re.M))
        if len(matches) != 5:
            return None
        for index, (step, match, phase) in enumerate(zip(steps, matches, phases, strict=True), 1):
            if (
                not isinstance(step, dict)
                or step.get("id") != index
                or step.get("phase") != phase
                or step.get("text") != match.group(2)
                or match.group(1) != str(index)
                or step.get("skip_if_fact") not in {None, "wired_connection", "billing_status"}
            ):
                return None
            if phase == "conditional_fix" and not (
                step["text"].startswith("Only if support confirms")
                and quotes["condition"].text in step["text"]
                and quotes["action"].text in step["text"]
            ):
                return None
            if phase == "completion" and quotes["completion"].text not in step["text"]:
                return None
            quotes[f"step_{index}"] = SourceQuote(
                field="plan_step",
                step_id=index,
                phase=phase,
                skip_if_fact=step.get("skip_if_fact"),
                text=match.group(2),
                start=match.start(2),
                end=match.end(2),
            )
    return Procedure(
        scope=quotes["scope"].text, quotes=quotes, baseline_for=result.metadata.get("baseline_for")
    )


def supported_scopes(analysis):
    """Use explicitly mentioned service families, without hard-filtering by predicted intent."""
    scopes = set().union(*(PRODUCT_SCOPES.get(item.product, set()) for item in analysis.products))
    if "iptv" in scopes and any(f.name == "tv_symptom" for f in analysis.reported_facts):
        return {"iptv"}
    affected = {f.value for f in analysis.reported_facts if f.name == "mobile_services"}
    mobile_scopes = PRODUCT_SCOPES["mobile"]
    if (
        affected
        and affected <= {"texts", "calls", "data"}
        and analysis.category not in {"mobile_coverage", "number_porting", "sim_esim_activation"}
    ):
        scopes -= mobile_scopes
        mapping = {"texts": "mobile_sms", "calls": "mobile_voice", "data": "mobile_data"}
        scopes.update(mapping[value] for value in affected)
        if analysis.category == "roaming":
            scopes -= {"mobile_data", "mobile_voice", "mobile_sms"}
            scopes.add("mobile_roaming")
    wired = {f.value for f in analysis.reported_facts if f.name == "wired_connection"}
    if wired == {"working"} and analysis.severity.rule != "reported_area_outage":
        scopes -= {"fibre_broadband", "router"}
        if any(p.product in {"broadband", "home_wifi", "router"} for p in analysis.products):
            scopes.add("home_wifi")
    return scopes
