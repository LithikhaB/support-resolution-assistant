"""Conservative English rules for explicit impact, tone, products and prior actions."""

import re

from app.understanding.models import (
    ActionObservation,
    ProductObservation,
    RuleAssessment,
    TextEvidence,
)

PRODUCT_PATTERNS = {
    "broadband": r"\b(?:broadband|fibre|fiber|ONT|Ethernet)\b",
    "home_wifi": r"\bwi[ -]?fi\b",
    "router": r"\b(?:router|gateway)\b",
    "mobile": r"\b(?:mobile|phone|handset|SIM|eSIM|roaming)\b",
    "iptv": r"\b(?:IPTV|TV|television|set.top box)\b",
    "billing": r"\b(?:bill|invoice|payment|charge|refund)\b",
}
TONE_PATTERNS = {
    "angry": r"\b(?:angry|furious|unacceptable|outraged)\b",
    "frustrated": r"\b(?:frustrated|fed up|tiresome|frustrating)\b",
    "concerned": r"\b(?:worried|worries|anxious|uneasy|concerned)\b",
    "neutral": r"\b(?:calmly|please explain|please advise|reporting the observations)\b",
}
ACTION_PATTERNS = {
    "restart_device": r"\b(?:restart(?:ed|ing)?|reboot(?:ed|ing)?|power[ -]cycl(?:e|ed|ing))\b",
    "test_wired_connection": r"\b(?:test(?:ed|ing)?|tried|check(?:ed|ing)?)\s+(?:(?:a|the|my)\s+)?(?:wired|Ethernet)\b",
    "check_cables": r"\b(?:check(?:ed|ing)?|reseat(?:ed|ing)?|swapp(?:ed|ing))\s+(?:(?:the|my|a)\s+)?(?:cables?|connectors?)\b",
    "compare_devices": r"\b(?:compar(?:ed|ing)|cross[ -]test(?:ed|ing)?)\b",
    "move_router": r"\b(?:mov(?:ed|ing)|reposition(?:ed|ing)?)\s+(?:(?:the|my|a)\s+)?router\b",
    "reset_settings": r"\breset\s+(?:(?:the|my)\s+)?(?:network|router|device|settings)\b",
}


def span(text: str, match: re.Match, offset: int = 0) -> TextEvidence:
    """Return exact character offsets into the normalized complaint."""
    return TextEvidence(text=match.group(), start=offset + match.start(), end=offset + match.end())


def negated(prefix: str) -> bool:
    """Recognize short local negations without treating failure after an action as negation."""
    return bool(
        re.search(
            r"\b(?:not|never|no|haven['’]t|hasn['’]t|didn['’]t|don['’]t|cannot|can['’]t)\b(?:\s+\w+){0,3}\s*$",
            prefix,
            re.I,
        )
    )


def extract_products(text: str) -> list[ProductObservation]:
    """Return the first explicit mention of each supported service or equipment family."""
    results = []
    for product, pattern in PRODUCT_PATTERNS.items():
        match = re.search(pattern, text, re.I)
        if match:
            results.append(ProductObservation(**span(text, match).model_dump(), product=product))
    return sorted(results, key=lambda item: item.start)


def assess_sentiment(text: str) -> RuleAssessment:
    """Report explicit English tone markers and leave unrecognized tone unknown."""
    for value, pattern in TONE_PATTERNS.items():
        for match in re.finditer(pattern, text, re.I):
            if not negated(text[max(0, match.start() - 40) : match.start()]):
                return RuleAssessment(
                    value=value, rule="explicit_tone", evidence=[span(text, match)]
                )
    return RuleAssessment(value="unknown", rule="no_explicit_tone")


def assess_severity(text: str) -> RuleAssessment:
    """Use reported service impact rather than sentiment; ambiguous impact stays unknown."""
    clauses = list(re.finditer(r"[^.!?;]+", text))
    area_pattern = r"\b(?:whole (?:street|area)|our street|neighbou?ring blocks|multiple buildings|regional|several neighbou?rs)\b"
    loss_pattern = (
        r"\b(?:lost|loss|outage|offline|no (?:internet|signal|service)|down|disconnected)\b"
    )
    for clause in clauses:
        area = re.search(area_pattern, clause.group(), re.I)
        loss = re.search(loss_pattern, clause.group(), re.I)
        if area and loss and not negated(clause.group()[: loss.start()]):
            return RuleAssessment(
                value="critical",
                rule="reported_area_outage",
                evidence=[span(text, area, clause.start()), span(text, loss, clause.start())],
            )
    rules = [
        (
            "high",
            "reported_complete_loss",
            r"\b(?:every device is offline|all devices (?:are offline|lost internet)|no (?:internet|mobile signal|power)|cannot (?:make|receive) calls|every outgoing call fails|completely (?:down|stopped working)|still suspended)\b",
        ),
        (
            "medium",
            "reported_degradation_or_work_impact",
            r"\b(?:drops?|disconnects?|buffering|slow|unstable|poor|weak signal|affecting my work|cannot work|can[’\x27]t work)\b",
        ),
        (
            "low",
            "reported_service_working",
            r"\b(?:every service works|all services work|service (?:is|remains) working)\b",
        ),
    ]
    for value, rule, pattern in rules:
        for match in re.finditer(pattern, text, re.I):
            if not negated(text[max(0, match.start() - 35) : match.start()]):
                return RuleAssessment(value=value, rule=rule, evidence=[span(text, match)])
    return RuleAssessment(value="unknown", rule="insufficient_impact_evidence")


def extract_actions(text: str) -> list[ActionObservation]:
    """Keep suggested and explicitly unperformed actions distinct from completed attempts."""
    results = []
    for clause in re.finditer(r"(?:(?!\bbut\b)[^.!?;,])+", text, re.I):
        fragment = clause.group()
        for action, pattern in ACTION_PATTERNS.items():
            for match in re.finditer(pattern, fragment, re.I):
                prefix = fragment[: match.start()]
                status = "attempted"
                if negated(prefix) or re.search(
                    r"\b(?:have not|haven[’\x27]t|never)\b[^;.!?]*$", prefix, re.I
                ):
                    status = "not_attempted"
                elif re.search(
                    r"\b(?:told|asked|advised|suggested|should|please|will|if|would|could|plan to|going to|can I)\b",
                    prefix,
                    re.I,
                ):
                    status = "suggested"
                elif match.group().lower() in {
                    "restart",
                    "reboot",
                    "power cycle",
                } and not re.search(r"\b(?:did|already|tried)\b", prefix, re.I):
                    status = "suggested"
                start = clause.start() + len(fragment) - len(fragment.lstrip())
                end = clause.start() + len(fragment.rstrip())
                results.append(
                    ActionObservation(
                        action=action, status=status, text=text[start:end], start=start, end=end
                    )
                )
    return sorted(results, key=lambda item: (item.start, item.action))
