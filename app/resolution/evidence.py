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
    if result.metadata.get("procedure_version") == 2:
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
    return Procedure(scope=quotes["scope"].text, quotes=quotes)


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
