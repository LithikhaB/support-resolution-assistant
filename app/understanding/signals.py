"""Conservative English rules for explicit impact, tone, products and prior actions."""

import re

from app.understanding.models import (
    ActionObservation,
    ProductObservation,
    RuleAssessment,
    TextEvidence,
)

PRODUCT_PATTERNS = {
    "broadband": r"\b(?:broadband|internet|web|fibre|fiber|ONT|optical box|LOS|Ethernet|wired (?:connection|computers?|devices?|tests?|PC))\b",
    "landline": r"\b(?:landline|home phone|phone wiring)\b",
    "home_wifi": r"\bwi[ -]?fi\b",
    "router": r"\b(?:router|gateway|modem|access point)\b",
    "mobile": r"\b(?:mobile|phones?(?!\s+(?:number|support))|handsets?|SIM|eSIM|roaming|SMS|OTP|calls?|text messages?|login text|travel pack|number (?:transfer|porting))\b",
    "iptv": r"\b(?:IPTV|TV|television|set.top box|live channels?)\b",
    "billing": r"\b(?:bills?|invoices?|payments?|charges?|refunds?|paid|account.*suspended)\b",
}
BUSINESS_IMPACT = r"\b(?:costing me(?: orders| money| business)?|losing (?:orders|income|customers)|lost (?:orders|income)|cannot work|can[’\x27]t work|affecting my (?:work|business))\b"

TONE_PATTERNS = {
    "angry": r"\b(?:angry|furious|unacceptable|outraged)\b",
    "frustrated": r"\b(?:frustrated|fed up|tiresome|frustrating|costing me(?: orders| money| business)?|losing (?:orders|income|customers)|restart\s+panni(?:tt|t)en\s+no use|cancel(?:ling|ing)\s+(?:my|the)\s+(?:connection|service|subscription))\b",
    "concerned": r"\b(?:worried|worries|anxious|uneasy|concerned)\b",
    "neutral": r"\b(?:calmly|please explain|please advise|reporting the observations)\b",
}
ACTION_PATTERNS = {
    "replace_power_adapter": r"\b(?:replaced|swapped|changed)\s+(?:(?:my|the|a|approved|ONT|router)\s+){0,3}(?:power\s+)?(?:adapter|adaptor|power supply)\b",
    "restart_device": r"\b(?:restart(?:ed|ing)?|reboot(?:ed|ing)?|power[ -]cycl(?:e|ed|ing)|unplugged\s+(?:it|(?:my|the)\s+(?:router|modem|ONT))\s+for\s+(?:\w+\s+){0,2}(?:minutes?|seconds?))\b",
    "test_wired_connection": r"\b(?:test(?:ed|ing)?|tried|check(?:ed|ing)?)\s+(?:(?:a|the|my)\s+)?(?:wired|Ethernet)\b",
    "check_cables": r"\b(?:check(?:ed|ing)?|reseat(?:ed|ing)?|swapp(?:ed|ing))\s+(?:(?:the|my|a|every|all)\s+)?(?:cables?|connectors?)\b",
    "compare_devices": r"\b(?:compar(?:ed|ing)|cross[ -]test(?:ed|ing)?)\b",
    "move_router": r"\b(?:mov(?:ed|ing)|reposition(?:ed|ing)?)\s+(?:(?:the|my|a)\s+)?router\b",
    "reset_settings": r"\b(?:reset\s+(?:(?:the|my)\s+)?(?:network|router|device|settings)|reset\s+(?:it\s+)?to\s+factory\s+settings|factory[ -]reset(?:\s+(?:it|the router))?)\b",
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
        matches = list(re.finditer(pattern, text, re.I))
        if product in {"broadband", "home_wifi", "router"} and re.search(
            PRODUCT_PATTERNS["billing"], text, re.I
        ):
            matches = [match for match in matches if not working_service_clause(text, match)]
        match = matches[0] if matches else None
        if match:
            results.append(ProductObservation(**span(text, match).model_dump(), product=product))
    products = {item.product for item in results}
    if (
        "billing" in products
        and re.search(r"\b(?:TV|television|IPTV)\s+(?:package|add.on|subscription)\b", text, re.I)
        and not re.search(
            r"\b(?:no picture|buffers?|buffering|freez\w*|channels? (?:missing|unavailable)|TV (?:fails?|does not work))\b",
            text,
            re.I,
        )
    ):
        results = [item for item in results if item.product != "iptv"]
    if "landline" in products:
        results = [item for item in results if item.product != "mobile"]
    if "home_wifi" in products:
        if not re.search(r"\b(?:SIM|SMS|OTP|roaming|mobile|calls?|cellular)\b", text, re.I):
            results = [item for item in results if item.product != "mobile"]
        if not re.search(r"\b(?:IPTV|channels?|set.top box|TV service)\b", text, re.I):
            results = [item for item in results if item.product != "iptv"]
    return sorted(results, key=lambda item: item.start)


def working_service_clause(text: str, match: re.Match) -> bool:
    """Recognize an explicitly working comparator without suppressing reported failures."""
    start = max(text.rfind(mark, 0, match.start()) for mark in ".!?;,") + 1
    ends = [text.find(mark, match.end()) for mark in ".!?;,"]
    end = min((position for position in ends if position >= 0), default=len(text))
    clause = text[start:end]
    return bool(
        re.search(r"\b(?:works?|working|fine|stable|okay)\b", clause, re.I)
    ) and not re.search(
        r"\b(?:not|no|isn't|isn’t|fails?|drops?|disconnects?|slow|offline|unavailable)\b",
        clause,
        re.I,
    )


def assess_sentiment(text: str) -> RuleAssessment:
    """Report explicit English tone markers and leave unrecognized tone unknown."""
    sarcasm = re.search(
        r"\b(?:oh brilliant|another lovely)[^.!?]{0,80}\b(?:no internet|outage|offline)\b",
        text,
        re.I,
    )
    if sarcasm:
        return RuleAssessment(
            value="frustrated", rule="sarcastic_service_complaint", evidence=[span(text, sarcasm)]
        )
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
    for clause in clauses:
        dependency = re.search(
            r"\b(?:medical alarm|emergency calls?|life.support|safety alarm)\b",
            clause.group(),
            re.I,
        )
        loss = re.search(
            r"\b(?:dead|down|not working|cannot|can't|unavailable|lost|no service)\b",
            clause.group(),
            re.I,
        )
        if (
            dependency
            and loss
            and text[clause.end() : clause.end() + 1] != "?"
            and not negated(clause.group()[: dependency.start()])
            and not re.search(r"\b(?:if|whether|might|maybe)\b", clause.group(), re.I)
        ):
            return RuleAssessment(
                value="critical",
                rule="reported_safety_dependency",
                evidence=[span(text, dependency, clause.start()), span(text, loss, clause.start())],
            )
    area_pattern = r"\b(?:(?:whole|entire) (?:street|area|road|block|lane)|(?:our|my|the) (?:street|road|lane|block)|neighbou?ring blocks|multiple buildings|regional|several neighbou?rs|neighbou?rs and I|every (?:shop|property|house|home|flat|building)s? (?:on|in|along) (?:our|the|this) (?:lane|road|street|block|area)|whole (?:lane|road|block))\b"
    loss_pattern = (
        r"\b(?:lost|loss|outage|offline|no (?:internet|signal|service)|down|disconnected)\b"
    )
    for clause in clauses:
        area = re.search(area_pattern, clause.group(), re.I)
        loss = re.search(loss_pattern, clause.group(), re.I)
        if (
            area
            and loss
            and text[clause.end() : clause.end() + 1] != "?"
            and not negated(clause.group()[: loss.start()])
            and not re.search(r"\b(?:if|whether|might|maybe|suppose)\b", clause.group(), re.I)
            and not negated(clause.group()[: area.start()])
        ):
            return RuleAssessment(
                value="critical",
                rule="reported_area_outage",
                evidence=[span(text, area, clause.start()), span(text, loss, clause.start())],
            )
    rules = [
        ("high", "reported_business_impact", BUSINESS_IMPACT),
        (
            "high",
            "reported_complete_loss",
            r"\b(?:neither[^.!?]{0,90}(?:can browse|can connect|works)|(?:all|every|both) (?:devices?|connections?)[^.!?]{0,40}(?:cannot|can't|can’t) (?:browse|connect)|(?:router|ONT|modem) (?:has |shows )?no lights|boot loop|keeps restarting|(?:broadband|internet|service|line)[^.!?]{0,25}(?:suspended|paused|cut off)|(?:no|lost|loss of) (?:optical )?signal)\b",
        ),
        (
            "medium",
            "reported_partial_service_or_billing_impact",
            r"\b(?:(?:two|duplicate|double) (?:completed |settled )?payments|charged twice|charged (?:me )?again|unauthori[sz]ed (?:charge|add.on)|(?:disputed|unexpected) (?:charge|line item|add.on)|(?:mobile data|data roaming) (?:does not|doesn't|is disabled|not working)|(?:number transfer|porting|activation) (?:was |is )?(?:rejected|stuck)|signal disappears|slowed|slowing|(?:live channels?|TV picture) (?:are |is )?(?:missing|unavailable|blank))\b",
        ),
        (
            "high",
            "reported_complete_loss",
            r"\b(?:no cellular browsing|cannot (?:reach|receive|send|dial|call)|will not register|turns? off|no dial tone|no (?:calls|texts|SMS)|(?:codes|messages|texts) (?:never arrive|do not arrive|don't arrive))\b",
        ),
        (
            "medium",
            "reported_degradation_or_work_impact",
            r"\b(?:stutter\w*|lags?|latency|unusable|barely works|not authori[sz]ed|activation (?:failed|stuck)|(?:speed|rate).{0,20}(?:below|only|stops? at))\b",
        ),
        (
            "low",
            "reported_billing_question",
            r"\b(?:(?:invoice|bill).{0,40}(?:higher|wrong|unexpected|part.month|dispute)|(?:duplicate|unexpected|incorrect|disputed) (?:charges?|payments?)|late (?:payment )?fee)\b",
        ),
        (
            "high",
            "reported_complete_loss",
            r"\b(?:(?:broadband|internet|mobile data|connection) (?:is |has been )?(?:not working|unavailable|offline|down)|(?:unable to|cannot|can't) (?:connect to the internet|access the internet|send (?:texts|SMS)|receive (?:texts|SMS))|(?:incoming|outgoing) calls (?:are )?(?:failing|blocked))\b",
        ),
        (
            "high",
            "reported_complete_loss",
            r"\b(?:every device is offline|all devices (?:are offline|lost internet)|no (?:internet|mobile signal|power)|cannot (?:make|receive) calls|every outgoing call fails|completely (?:down|stopped working)|still suspended|(?:line|service) (?:remains|is) suspended|calls? fail(?:s)?(?: immediately)?|cannot get online|can[’\x27]t get online|shuts? down|overheat(?:s|ing)?|(?:red|blinking) LOS(?: light)?|LOS (?:light )?(?:is |shows |stays )?red|no received (?:optical )?signal)\b",
        ),
        (
            "medium",
            "reported_degradation_or_work_impact",
            r"\b(?:packet loss|frequent disconnections|keeps disconnecting|intermittent (?:internet|connection|broadband)|video (?:freezes|stutters)|speed (?:has )?dropped)\b",
        ),
        (
            "medium",
            "reported_degradation_or_work_impact",
            r"\b(?:drops?|cuts? out|vanish(?:es|ing)|freez(?:es|ing)|disconnects?|disconnecting|buffering|slow|unstable|poor|weak signal|no service|no picture|activation.*pending|people calling me reach an error|affecting my work|cannot work|can[’\x27]t work)\b",
        ),
        (
            "low",
            "reported_service_working",
            r"\b(?:every service works|all services work|service (?:is|remains) working)\b",
        ),
    ]
    for value, rule, pattern in rules:
        for match in re.finditer(pattern, text, re.I):
            prefix = re.split(r"[.!?;]", text[: match.start()])[-1]
            suffix = text[match.end() :]
            sentence_end = re.search(r"[.!?;]", suffix)
            is_question = sentence_end is not None and sentence_end.group() == "?"
            if (
                not is_question
                and not negated(prefix[-35:])
                and not re.search(r"\b(?:if|whether|might|maybe|suppose)\b", prefix, re.I)
            ):
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
                elif (
                    match.group().lower()
                    in {
                        "restart",
                        "reboot",
                        "power cycle",
                    }
                    and not re.search(r"\b(?:did|already|tried)\b", prefix, re.I)
                    and not re.match(r"\s+panni(?:tt|t)en\b", fragment[match.end() :], re.I)
                ):
                    status = "suggested"
                start = clause.start() + len(fragment) - len(fragment.lstrip())
                conjunctions = list(re.finditer(r"\b(?:and|but|so)\s+", prefix, re.I))
                if conjunctions:
                    start = clause.start() + conjunctions[-1].end()
                end = clause.start() + len(fragment.rstrip())
                # A comma can separate duration from repetition, e.g. ten minutes, twice.
                repeat = re.match(r",\s*(?:twice|once|\d+ times|three times)\b", text[end:], re.I)
                if repeat:
                    end += repeat.end()
                results.append(
                    ActionObservation(
                        action=action, status=status, text=text[start:end], start=start, end=end
                    )
                )
    return sorted(results, key=lambda item: (item.start, item.action))
