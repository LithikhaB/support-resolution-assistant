"""Suppress repeated remedies while retaining the highest-ranked complete source."""

import re

from app.understanding.signals import negated


def action_key(procedure):
    """Normalize exact actions and the KB's narrowly defined network-capacity remedy."""
    action = procedure.quotes["action"].text
    normalized = " ".join(re.findall(r"[a-z0-9]+", action.casefold()))
    capacity = re.search(r"\bcapacity\b", action, re.I)
    change = re.search(r"\b(?:expand|rebalance|increase)\b", action, re.I)
    if (
        capacity
        and change
        and re.search(r"\bnetwork operations\b", action, re.I)
        and not negated(action[: change.start()])
    ):
        normalized = "network_capacity_adjustment"
    return procedure.scope, normalized
