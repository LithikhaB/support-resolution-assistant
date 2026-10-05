"""Conservative explicit scope detection and separation of distinct service complaints."""

import re

from app.understanding.signals import extract_products, negated

UNSUPPORTED = re.compile(
    r"\b(?:(?:give|show|find) me (?:a |the )?recipe|bake a cake|write (?:a |an )?(?:poem|essay)|(?:do|solve) my homework|"
    r"(?:give|show|tell) me (?:the )?(?:stock (?:price|advice)|weather forecast)|book a flight)\b",
    re.I,
)

NON_COMPLAINT_REQUEST = re.compile(
    r"^\s*(?:please\s+)?(?:paste|write|generate|create)\s+"
    r"(?:(?:me|a|an|any|the)\s+)*(?:\d+[ -]word\s+)?"
    r"(?:single[ -]topic\s+)?(?:(?:telecom|broadband|mobile|billing|customer)\s+)?"
    r"complaint\b",
    re.I,
)
GENERAL_KNOWLEDGE = re.compile(
    r"\b(?:who\s+(?:is|was|are)\s+(?:the\s+)?(?:current\s+)?"
    r"(?:prime minister|president)|what\s+is\s+the\s+capital\s+of)\b",
    re.I,
)


def explicitly_unsupported(text):
    """Require a direct unrelated request and exclude locally negated requests."""
    return bool(NON_COMPLAINT_REQUEST.search(text)) or any(
        not negated(re.split(r"[.!?;]", text[: match.start()])[-1])
        for pattern in (UNSUPPORTED, GENERAL_KNOWLEDGE)
        for match in pattern.finditer(text)
    )


def scope_assessment(text):
    """Reject only recognizable unrelated requests; unknown wording remains uncertain."""
    if explicitly_unsupported(text):
        return "unsupported", "explicit_unrelated_request"
    if extract_products(text):
        return "supported", "recognized_service_mention"
    if re.search(
        r"^\s*(?:who|what|where|when|why|how|explain|tell me)\b", text, re.I
    ) and not re.search(
        r"\b(?:disconnect\w*|drops?|connection|signal|service|charge[ds]?|payment|invoice|offline|outage|buffer\w*|activate|speed|coverage|reconnect\w*)\b",
        text,
        re.I,
    ):
        return "unsupported", "question_without_telecom_issue"
    return "uncertain", "service_not_identified"


def service_group(text):
    """Collapse router, Wi-Fi and broadband mentions into one connectivity issue."""
    products = {p.product for p in extract_products(text)}
    if "landline" in products:
        products.discard("mobile")
    if products & {"home_wifi", "broadband", "router"}:
        if not re.search(r"\b(?:SIM|SMS|OTP|roaming|mobile data|calls?|cellular)\b", text, re.I):
            products.discard("mobile")
        if not re.search(r"\b(?:IPTV|set.top box|channels?|TV service)\b", text, re.I):
            products.discard("iptv")
    groups = products - {"broadband", "home_wifi", "router"}
    if products & {"broadband", "home_wifi", "router"}:
        groups.add("home_connectivity")
    return groups


def working_comparison(text):
    """Keep a clearly working comparison device with the complaint it helps explain."""
    return bool(
        re.search(r"\b(?:works?|working|fine|normally|okay|OK)\b", text, re.I)
    ) and not re.search(
        r"\b(?:not|no|isn't|isn’t|fails?|drops?|disconnects?|missing|dead|slow|freez\w*|buffers?)\b",
        text,
        re.I,
    )


def followup_groups(text, original):
    """Allow a billing answer to name its service without accepting another technical fault."""
    groups = service_group(text)
    if (
        original == {"billing"}
        and "billing" in groups
        and not re.search(
            r"\b(?:offline|outage|down|disconnect\w*|no (?:signal|service|data|picture)|not working|does(?:n't| not) work|cannot (?:activate|call)|won't activate|slow|buffers?|freez\w*)\b",
            text,
            re.I,
        )
    ):
        return {"billing"}
    return groups


def split_issues(text):
    """Split explicit service changes while retaining unlabelled details with their issue."""
    if re.search(r"\b(?:floodwater|flood waters?|water.damage)\b", text, re.I) and service_group(
        text
    ) <= {"home_connectivity", "landline"}:
        return [text]
    fragments = re.split(
        r"(?<=[.!?;])\s+|\s+(?:and also|also|plus|but)\s+|\s+and\s+(?=(?:my|the)\s)",
        text,
        flags=re.I,
    )
    issues = []
    previous = set()
    for fragment in fragments:
        fragment = fragment.strip()
        if not fragment:
            continue
        groups = service_group(fragment)
        if issues and working_comparison(fragment):
            issues[-1] += " " + fragment
            continue
        if (
            "iptv" in previous
            and groups <= {"iptv", "home_connectivity"}
            and re.search(r"\b(?:set.top box|TV|television)\b", fragment, re.I)
        ):
            groups = previous
        if (
            "home_connectivity" in previous
            and groups <= {"mobile", "iptv"}
            and not re.search(
                r"\b(?:SIM|SMS|OTP|roaming|mobile data|calls?|cellular|IPTV|channels?|set.top box)\b",
                fragment,
                re.I,
            )
        ):
            groups = {"home_connectivity"}
        if explicitly_unsupported(fragment):
            groups = {"unsupported"}
        if issues and (not groups or not previous or groups == previous or groups <= previous):
            issues[-1] += " " + fragment
            previous = groups or previous
        else:
            issues.append(fragment)
            previous = groups
    return issues or [text]
