"""Extract reported observations and choose questions from the available context."""

import re

from app.understanding.models import CustomerRequest, ReportedFact
from app.understanding.signals import negated, span

AREA_OUTAGE_QUESTION = "Which area is affected, and when did the shared outage begin? Do not delay incident review while collecting these details."

EQUIPMENT = r"(?:modem|router|ont|equipment|cables?|landlines?|ethernet line)"
EQUIPMENT_LINK = r"(?:\s+(?:is|are|was|were|has|have|been|got|looks?|seems?|everything|all|physically|completely|badly|casing)){0,6}\s+"

FACT_PATTERNS = (
    (
        "weather_context",
        "reported",
        r"\b(?:storm|rain|raining|wet weather|thunderstorm|lightning)\b",
    ),
    ("timing", "peak_hours", r"\b(?:evenings?|peak[ -]hours?|busy hours?|nightly)\b"),
    (
        "wired_connection",
        "working",
        r"\b(?:Ethernet|wired(?: connection)?)\s+(?:stays|remains)\s+(?:working|connected|stable)\b",
    ),
    ("wireless_devices", "all", r"\b(?:every|all)\s+wireless\s+devices?\b"),
    (
        "port_status",
        "rejected",
        r"\b(?:number transfer|port(?:ing)?(?: request| order)?)\s+(?:was |is |has been )?rejected\b",
    ),
    (
        "port_reason",
        "account_mismatch",
        r"\bdetails\s+(?:do not|don't|don’t)\s+match\b|\baccount\s+(?:details?\s+)?mismatch\b",
    ),
    (
        "security_request",
        "otp_sharing",
        r"\b(?:read out|share|give|send|tell)[^.!?]{0,40}\b(?:OTP|verification code|one.time (?:password|code))\b",
    ),
    ("started", "reported", r"\b(?:at|since|from)\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)\b"),
    (
        "optical_signal",
        "normal_reported",
        r"\b(?:optical|PON|LOS)\s+(?:light|signal)\s+(?:is|remains|stays)\s+(?:normal|green)\b",
    ),
    (
        "equipment_condition",
        "water_exposed",
        r"\bwater\s+(?:got|came)\s+into\s+(?:my|the)\s+(?:broadband\s+)?(?:router|modem|ONT|equipment)\b",
    ),
    ("charge", "late_fee", r"\blate\s+(?:payment\s+)?fee\b"),
    (
        "bill_status",
        "unpaid",
        r"\b(?:will|I'll|I’ll)\s+(?:pay|settle)\b[^.!?;]{0,35}\bbill\b|\bbill\s+(?:is\s+)?(?:still\s+)?(?:unpaid|pending)\b|\b(?:haven't|have not)\s+paid\s+(?:my|the)\s+bill\b",
    ),
    (
        "optical_signal",
        "loss_reported",
        r"\b(?:red\s+LOS(?:\s+light)?|LOS\s+(?:light\s+)?(?:is\s+)?red|(?:optical box|ONT)\s+(?:has|shows)\s+(?:a\s+)?red\s+light)\b",
    ),
    (
        "wired_connection",
        "failing",
        r"\b(?:wired|Ethernet)\s+(?:PC|desktop|laptop|computer|device)\s+(?:\w+\s+){0,3}and\s+it\s+(?:also\s+|still\s+)?(?:loses? internet|drops?|disconnects?)\b",
    ),
    (
        "wired_connection",
        "failing",
        r"\b(?:wired|Ethernet|cable.connected)\s+(?:PC|desktop|laptop|computer|device|connection)\s+(?:(?:still|also|keeps?|is)\s+)*(?:drops?|disconnects?|loses? (?:internet|connection)|offline|cuts? out)\b|\b(?:laptop|desktop|PC|computer|connection)\s+(?:(?:still|also|keeps?)\s+)*(?:drops?|disconnects?|cuts? out)\s+(?:over|via|on)\s+Ethernet\b",
    ),
    (
        "wired_connection",
        "failing",
        r"\b(?:a\s+device\s+connected\s+by\s+Ethernet\s+also\s+loses\s+internet|device\s+on\s+Ethernet\s+also\s+loses\s+connection|Ethernet\s+also\s+loses\s+internet)\b",
    ),
    (
        "wired_connection",
        "unavailable",
        r"\b(?:no\s+(?:ethernet|wired)\s+(?:cable|device|port|connection)?|cannot\s+(?:do|use|test)\s+ethernet|can't\s+(?:do|use|test)\s+ethernet)\b",
    ),
    (
        "impact",
        "complete_loss",
        r"\b(?:all\s+devices\s+are\s+shut\s+down|service\s+is\s+completely\s+lost|completely\s+lost|all\s+devices\s+lost\s+internet)\b",
    ),
    (
        "wired_connection",
        "working",
        r"\b(?:wired|Ethernet|cable.connected)\s+(?:PC|desktop|laptop|computer|device|connection)\s+(?:stays? online|works?(?: fine)?|is (?:fine|stable)|remains? connected)\b|\bEthernet\s+works?\s+fine\s+on\s+(?:my|the)\s+(?:desktop|laptop|PC|computer)\b",
    ),
    (
        "wired_connection",
        "unavailable",
        r"\b(?:can't|cannot)\s+(?:try|test)\s+Ethernet\b",
    ),
    (
        "wired_connection",
        "failing",
        r"\b(?:with|using|via|over)\s+(?:an?\s+)?(?:Ethernet\s+)?cable\s+(?:and\s+)?(?:it|the connection)\s+(?:still\s+|also\s+)?(?:drops|disconnects|cuts out|fails)\b",
    ),
    (
        "mobile_services",
        "data",
        r"\b(?:have|has|getting)\s+no\s+(?:mobile\s+)?data\b|\b(?:mobile\s+)?data\s+(?:isn't|is not|doesn't|does not)\s+work(?:ing)?\b",
    ),
    (
        "mobile_services",
        "texts",
        r"\b(?:can't|cannot|can not)\s+(?:get|receive)\b[^.!?;,]{0,40}\b(?:one.time passwords?|verification (?:codes?|texts?)|SMS|OTP)\b",
    ),
    (
        "sms_scope",
        "one_sender",
        r"\bonly\s+(?:the\s+)?[\w'’ -]{1,35}\b(?:ones|messages|codes|texts)\s+(?:are\s+)?(?:missing|not arriving)\b",
    ),
    (
        "tv_symptom",
        "buffering",
        r"\b(?:live channels?|TV|television|picture|set.top box)\s+(?:(?:keeps?|is|are|still)\s+)*(?:freez(?:e|es|ing)|buffers?|buffering)\b",
    ),
    (
        "charge",
        "bill_payment",
        r"\b(?:two|duplicate)\s+(?:(?:completed|settled)\s+)?(?:charges|payments)\b",
    ),
    (
        "wired_connection",
        "unavailable",
        r"\b(?:cannot|can't)\s+(?:plug|use|test|connect)[^.!?]{0,60}\bEthernet\b|\b(?:don't|do not) have\s+(?:any\s+)?devices? with Ethernet\b",
    ),
    (
        "equipment_condition",
        "water_exposed",
        r"\bfloodwater\s+(?:got|came)\s+into\s+(?:my|the)\s+(?:modem|router|equipment)\b",
    ),
    (
        "billing_status",
        "pending",
        r"\b(?:money|payments?|transactions?)\b(?:(?!\b(?:not|no)\b)[^.!?]){0,35}\bpending\b",
    ),
    ("billing_status", "settled", r"\b(?:both|two)[^.!?]{0,40}\b(?:settled|completed)\b"),
    ("charge", "bill_payment", r"\b(?:paid my bill|same(?: monthly)? bill|duplicate charges?)\b"),
    (
        "mobile_services",
        "texts",
        r"\b(?:banking verification texts|verification texts|OTP|SMS)[^.!?]{0,35}\b(?:never arrive|missing|not arriving)\b",
    ),
    (
        "wireless_devices",
        "one",
        r"\b(?:my |the )?phone is the only (?:thing|device) that disconnects\b",
    ),
    (
        "equipment_condition",
        "water_exposed",
        rf"\b{EQUIPMENT}\b{EQUIPMENT_LINK}(?:soaked|submerged|water[ -]damaged|flood[ -]damaged|wet)\b",
    ),
    (
        "equipment_condition",
        "damaged",
        rf"\b{EQUIPMENT}\b{EQUIPMENT_LINK}(?:damaged|broken|frayed|burnt|burned)\b|\b(?:damaged|broken|frayed|burnt|burned)\s+{EQUIPMENT}\b",
    ),
    (
        "equipment_condition",
        "intact",
        r"\b(?:modem|router|ont|equipment)\s+(?:is|are)\s+(?:undamaged|intact|not damaged|not wet)\b",
    ),
    (
        "wired_connection",
        "failing",
        r"\b(?:Ethernet|wired(?: connection)?)\s+(?:also\s+)?(?:drops|disconnects|fails|is offline|does not work|doesn't work)\b",
    ),
    (
        "wired_connection",
        "failing",
        r"\ba device connected by Ethernet\s+(?:also\s+)?(?:loses? (?:internet|connection)|drops?|disconnects?|is offline|cuts? out)\b",
    ),
    (
        "wired_connection",
        "failing",
        r"\bdevice(?:s)?\s+(?:on|via|using|connected (?:by|via|with))\s+Ethernet\s+(?:also\s+)?(?:loses? (?:internet|connection)|drops?|disconnects?|is offline|has no internet)\b",
    ),
    (
        "wired_connection",
        "failing",
        r"\bEthernet[- ]connected device\s+(?:is\s+)?(?:also\s+)?(?:offline|losing internet|disconnected|has no internet|drops?)\b",
    ),
    (
        "wireless_devices",
        "all",
        r"\b(?:all|every)\s+(?:wireless|wi[ -]?fi)\s+devices?\s+(?:are affected|is affected|disconnect|disconnects|drop|drops)\b",
    ),
    (
        "wireless_devices",
        "one",
        r"\b(?:only one|a single)\s+(?:wireless|wi[ -]?fi)\s+device\s+(?:is affected|disconnects|drops)\b",
    ),
    (
        "service_recovery",
        "working",
        r"\b(?:all services|everything)\s+(?:is |are )?(?:working|works)(?: again| now)?\b",
    ),
    (
        "cable_condition",
        "intact",
        r"\bcables?(?:\s+(?:in|at)\s+(?:(?:my|the)\s+)?(?:home|house))?\s+(?:is|are|looks?|seems?)\s+(?:physically\s+)?(?:intact|undamaged|fine)\b",
    ),
    (
        "cable_condition",
        "damaged",
        r"\bcables?(?:\s+(?:in|at)\s+(?:(?:my|the)\s+)?(?:home|house))?\s+(?:is|are|looks?|seems?)\s+(?:damaged|broken|frayed)\b",
    ),
    (
        "wired_connection",
        "working",
        r"\b(?:Ethernet|wired(?: connection)?)\s+(?:is\s+)?(?:works?|working|stable|fine)\b",
    ),
    (
        "connection_pattern",
        "intermittent",
        r"\b(?:broadband|internet|connection|wi[ -]?fi)\s+(?:keeps?\s+)?(?:dropping|drops|disconnects|disconnecting|cuts out)\b",
    ),
)


def extract_facts(text: str) -> list[ReportedFact]:
    """Keep explicit positive statements, excluding questions, conditions and negations."""
    facts = []
    for clause in re.finditer(r"[^.!?;,]+[.!?;,]?", text):
        fragment = clause.group()
        if fragment.rstrip().endswith("?") or re.search(
            r"\b(?:if|whether|suppose|might|maybe)\b", fragment, re.I
        ):
            continue
        for name, value, pattern in FACT_PATTERNS:
            for match in re.finditer(pattern, fragment, re.I):
                if negated(fragment[: match.start()]):
                    continue
                if (
                    name == "equipment_condition"
                    and value != "intact"
                    and re.search(r"\b(?:not|never|no|isn't|aren't)\b", match.group(), re.I)
                ):
                    continue
                fact_value = value
                sentence_start = max(text.rfind(mark, 0, clause.start()) for mark in ".!?") + 1
                context = text[sentence_start : clause.end()]
                if (
                    name == "equipment_condition"
                    and value == "damaged"
                    and re.search(
                        r"\b(?:flood|floods|floodwater|flood waters?|water damage)\b", context, re.I
                    )
                    and not re.search(r"\b(?:not|no|if|maybe|might)\b.{0,25}\bflood", context, re.I)
                ):
                    fact_value = "water_exposed"
                facts.append(
                    ReportedFact(
                        **span(text, match, clause.start()).model_dump(),
                        name=name,
                        value=fact_value,
                    )
                )
    return sorted(facts, key=lambda item: (item.start, item.name))


def extract_requests(text: str) -> list[CustomerRequest]:
    """Recognize requests for contact details and further help using explicit wording."""
    requests = []
    patterns = {
        "contact_support": r"\b(?:helpline(?: number)?|toll[ -]?free(?: number)?|customer care(?: number)?|support (?:phone |contact )?number|contact (?:support|an agent|a human)|speak to (?:an agent|a human)|call (?:support|an agent|a human|my provider))\b",
        "next_steps": r"\b(?:what (?:should I do|to do)(?: now| next)?|next steps?|what can I (?:do|try))\b",
        "replacement": r"\b(?:request|need|want|arrange)(?:\s+\w+){0,4}\s+replacement\b|\breplace (?:my|the) (?:modem|router|equipment|cable)\b",
    }
    for kind, pattern in patterns.items():
        for match in re.finditer(pattern, text, re.I):
            if not negated(text[max(0, match.start() - 45) : match.start()]):
                requests.append(CustomerRequest(**span(text, match).model_dump(), kind=kind))
    return sorted(requests, key=lambda item: item.start)


def clarification_questions(*, accepted, products, severity, facts, requests) -> list[str]:
    """Ask for missing diagnostic context rather than repeating known service details."""
    questions = []
    services = {item.product for item in products}
    known = {(item.name, item.value) for item in facts}
    if ("security_request", "otp_sharing") in known:
        return []
    if any(
        name == "equipment_condition" and value in {"damaged", "water_exposed"}
        for name, value in known
    ):
        return []
    if "billing" not in services and (
        ("service_recovery", "working") in known or ("impact", "working") in known
    ):
        return []
    if services == {"landline"}:
        return []
    if severity.rule == "reported_area_outage":
        questions.append(
            "Which area is affected? Do not delay incident review while collecting these details."
            if any(name == "started" for name, _ in known)
            else AREA_OUTAGE_QUESTION
        )
    else:
        if "iptv" in services and any(name == "tv_symptom" for name, _ in known):
            pass
        elif services & {"broadband", "home_wifi", "router"}:
            # If impact is already confirmed as complete loss, wired/wireless isolation
            # is not actionable — everything is down, no point asking about Ethernet.
            impact_complete = ("impact", "complete_loss") in known
            if impact_complete and any(name == "wired_connection" for name, _ in known):
                # Both impact and wired status are known — no connectivity question needed.
                pass
            elif ("wireless_devices", "one") in known:
                questions.append(
                    "Does the affected device reconnect by itself, or do you need to turn its Wi-Fi off and on?"
                )
            elif ("wired_connection", "unavailable") in known:
                if not any(name == "wireless_devices" for name, _ in known):
                    questions.append(
                        "Without using Ethernet, do all your Wi-Fi devices lose connection at the same time, or just one?"
                    )
            elif ("wired_connection", "working") in known:
                if not any(name == "wireless_devices" for name, _ in known):
                    questions.append(
                        "Does the Wi-Fi problem affect one device or every wireless device?"
                    )
            elif ("wired_connection", "failing") in known:
                # Wired is confirmed failing — no need to ask about it.
                pass
            elif impact_complete:
                # Complete loss reported but no wired observation yet — still relevant to ask.
                questions.append(
                    "During a drop, does a device connected by Ethernet also lose internet, or is only Wi-Fi affected?"
                )
            else:
                questions.append(
                    "During a drop, does a device connected by Ethernet also lose internet, or is only Wi-Fi affected?"
                )
        elif "billing" in services:
            fields = {name for name, _ in known}
            if ("bill_status", "unpaid") in known:
                pass
            elif ("charge", "late_fee") in known:
                questions.append(
                    "Was this bill paid before the due date, after it, or is it still unpaid?"
                )
            elif "billing_status" in fields and "charge" not in fields:
                questions.append(
                    "Which bill or charge do these payments relate to? Share a non-sensitive description; do not send bank credentials or card details."
                )
            elif "charge" in fields and "billing_status" not in fields:
                questions.append(
                    "Does your payment record show pending or settled? Do not share bank credentials or card details."
                )
            elif not {"charge", "billing_status"} <= fields:
                questions.append(
                    "Which charge or payment is affected, and is it pending or settled?"
                )
        elif "mobile" in services:
            if not any(name == "mobile_services" for name, _ in known):
                questions.append("Are calls, texts, mobile data, or several of these affected?")
        elif "iptv" in services:
            if not any(name == "tv_symptom" for name, _ in known):
                questions.append("Is the TV showing no picture, an error, or buffering?")
        elif not accepted:
            questions.append("Which service is affected, and what exactly happens when you use it?")
    if (
        severity.value == "unknown"
        and "billing" not in services
        and not any(name == "service_recovery" for name, _ in known)
        and not any(name == "impact" for name, _ in known)
        and not any(name == "mobile_services" for name, _ in known)
        and not any(name in {"connection_pattern", "tv_symptom"} for name, _ in known)
    ):
        questions.append(
            "Is service completely unavailable or intermittent, and which devices or people are affected?"
        )
    return questions
