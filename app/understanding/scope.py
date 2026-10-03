"""Conservative explicit scope detection and separation of distinct service complaints."""

import re

from app.understanding.signals import extract_products, negated

UNSUPPORTED = re.compile(
    r"\b(?:(?:give|show|find) me (?:a |the )?recipe|bake a cake|write (?:a |an )?(?:poem|essay)|(?:do|solve) my homework|"
    r"(?:give|show|tell) me (?:the )?(?:stock (?:price|advice)|weather forecast)|book a flight)\b",
    re.I,
)


def explicitly_unsupported(text):
    """Require a direct unrelated request and exclude locally negated requests."""
    return any(
        not negated(re.split(r"[.!?;]", text[: match.start()])[-1])
        for match in UNSUPPORTED.finditer(text)
    )


def scope_assessment(text):
    """Reject only recognizable unrelated requests; unknown wording remains uncertain."""
    if explicitly_unsupported(text):
        return "unsupported", "explicit_unrelated_request"
    if extract_products(text):
        return "supported", "recognized_service_mention"
    return "uncertain", "service_not_identified"


def service_group(text):
    """Collapse router, Wi-Fi and broadband mentions into one connectivity issue."""
    products = {p.product for p in extract_products(text)}
    groups = products - {"broadband", "home_wifi", "router"}
    if products & {"broadband", "home_wifi", "router"}:
        groups.add("home_connectivity")
    return groups


def split_issues(text):
    """Split explicit service changes while retaining unlabelled details with their issue."""
    fragments = re.split(
        r"(?<=[.!?;])\s+|\s+(?:and also|also|plus)\s+|\s+and\s+(?=(?:my|the)\s)", text, flags=re.I
    )
    issues = []
    previous = set()
    for fragment in fragments:
        fragment = fragment.strip()
        if not fragment:
            continue
        groups = service_group(fragment)
        if explicitly_unsupported(fragment):
            groups = {"unsupported"}
        if issues and (not groups or not previous or groups == previous):
            issues[-1] += " " + fragment
            previous = groups or previous
        else:
            issues.append(fragment)
            previous = groups
    return issues or [text]
