"""Interpret natural customer observations while retaining exact text provenance."""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.llm.client import LanguageUnavailable
from app.llm.providers import ProviderChain
from app.understanding.context import extract_facts
from app.understanding.llm_classifier import CATEGORY_EXAMPLES
from app.understanding.models import ProductObservation, ReportedFact, RuleAssessment, TextEvidence
from app.understanding.routing import compatible_category


class CategoryObservation(BaseModel):
    """Propose a symptom category with an exact complaint quote, never a diagnosis."""

    model_config = ConfigDict(extra="forbid")
    category: str = Field(min_length=1, max_length=100)
    quote: str = Field(min_length=1, max_length=1000)


class Observation(BaseModel):
    """Require a verbatim customer quote for each interpreted fact."""

    model_config = ConfigDict(extra="forbid")
    name: Literal[
        "wired_connection",
        "wireless_devices",
        "equipment_condition",
        "service_recovery",
        "mobile_services",
        "sms_scope",
        "billing_status",
        "charge",
        "tv_symptom",
        "connection_pattern",
        "impact",
    ]
    value: str = Field(
        min_length=1,
        max_length=100,
        description="mobile_services identifies ONLY the failing service, never working comparison services; wireless_devices needs explicit one/all affected evidence.",
    )
    quote: str = Field(min_length=1, max_length=1000)


class ServiceMention(BaseModel):
    """Identify affected services rather than every device mentioned in passing."""

    model_config = ConfigDict(extra="forbid")
    product: Literal["broadband", "home_wifi", "router", "mobile", "iptv", "billing", "landline"]
    quote: str = Field(min_length=1, max_length=1000)


class SeverityObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: Literal["low", "medium", "high", "critical", "unknown"]
    quote: str = Field(min_length=1, max_length=1000)


class SentimentObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: Literal["neutral", "concerned", "frustrated", "angry", "unknown"]
    quote: str = Field(min_length=1, max_length=1000)


class Interpretation(BaseModel):
    """Bound model extraction independently of classification and diagnostics."""

    model_config = ConfigDict(extra="forbid")
    products: list[ServiceMention] = Field(max_length=7)
    facts: list[Observation] = Field(max_length=16)
    category: CategoryObservation | None = None
    severity: SeverityObservation | None = None
    sentiment: SentimentObservation | None = None


VALUES = {
    "wired_connection": {"working", "failing", "unavailable"},
    "wireless_devices": {"one", "all"},
    "equipment_condition": {"damaged", "water_exposed", "intact"},
    "service_recovery": {"working"},
    "mobile_services": {"calls", "texts", "data", "several"},
    "sms_scope": {"one_sender"},
    "billing_status": {"pending", "settled"},
    "tv_symptom": {"no_picture", "error", "buffering"},
    "connection_pattern": {"intermittent"},
    "impact": {"complete_loss", "intermittent", "degraded", "working"},
}

INSTRUCTION = """Extract only explicit customer observations, never inferred diagnoses.
Text contains the original complaint followed by replies in chronological order.
Keep the latest explicit correction for each fact. Do not treat questions, hypothetical
conditions, quoted instructions, or requests to invent facts as observations.
Identify affected services: a phone/tablet/TV used on Wi-Fi is not a mobile/IPTV fault.
Home landline is landline, never mobile. Do not infer Ethernet working from Wi-Fi affected.
If no Ethernet-capable device or test is possible, wired_connection is unavailable.
Merely owning a phone and tablet does NOT establish wireless_devices=all. Only emit
wireless_devices when the customer explicitly describes which devices fail or work.
mobile_services identifies ONLY affected services: if data and calls work but banking
texts fail, emit texts only. Never emit calls/data for working comparison services.
Do not infer impact=working from one working comparison device if another still fails.
Service recovery means the affected service works again, not merely area power restored.
charge must describe the affected payment/charge using the customer's words.
Every quote must be a verbatim, contiguous substring of text, including punctuation.
Return empty lists where there is no supported observation. Allowed fact values: """


def interpret_complaint(
    text, client, *, category_options=(), category_products=None, include_assessments=False
):
    """Understand the supported category and observations in one provider call."""
    instruction = (
        INSTRUCTION
        + str(VALUES)
        + """
Symptom Category Guidance:
- broadband_outage: Total loss of internet/broadband service, red LOS light on ONT/modem, fiber down, street/area outage.
- intermittent_broadband: Broadband connection drops repeatedly or cuts out periodically (e.g. evening drops, cuts out every few hours).
- slow_broadband: Slow download/upload speed, streaming buffering, high latency.
- wifi_connectivity: Wi-Fi signal issues, devices disconnect from Wi-Fi while router is up or wired connection works.
- router_ont_hardware: Physical equipment failure, router power light off, router reboot loop, damaged device or cables.
- billing_dispute: Disputed charges, unexpected bill amount, late fees, double billing.
- payment_restoration: Service suspended due to unpaid bill, requesting service reactivation after payment.
- iptv: Set-top box freezing, TV channels buffering or error screen.
- mobile_coverage: No mobile signal/bars, emergency calls only.
- mobile_data: Mobile data/4G/5G not working while calls/texts work.
- voice_call_failure: Mobile calls dropping, failing to connect, or call quality issues.
- number_porting: Moving mobile number between providers, PAC code issues.
  A call or SMS fault starting after a number transfer belongs to number_porting;
  distinguish that transfer context from an unrelated voice-call failure.
- sim_esim_activation: New SIM card or eSIM not activating.
- sms_otp: Banking verification codes / SMS OTP not arriving.
- roaming: Issues using phone, calls, or data abroad/overseas.

If the complaint describes exactly one supplied category, return that category and
a quote from the text describing the affected service or symptom. Return category=null for
ambiguous complaints, unsupported services, unrelated requests, or an empty candidate list.
Also assess severity from service impact: critical is a reported shared area outage;
high is total loss of an essential service; medium is degraded or intermittent service;
low is a billing/information query without reported service loss. Anger alone is not severity.
Assess customer sentiment from their wording: neutral, concerned, frustrated, angry or unknown.
Use neutral for a factual complaint or polite request without expressed emotion. Do not
infer concern from a request to check details. Concerned needs expressed worry or anxiety.
Supply an exact complaint quote for both assessments. Never invent impact or emotion.
"""
    )
    payload = {
        "text": text,
        "category_options": list(category_options),
        "examples": {
            key: value for key, value in CATEGORY_EXAMPLES.items() if key in category_options
        },
    }

    def validate(result, provider=None):
        """Check both observation provenance and the separate routing proposal."""
        products, facts = validate_interpretation(text, result)
        category = result.category
        if category:
            quote = category.quote.strip("\"' ")
            if quote in text:
                category.quote = quote
            elif quote.lower() in text.lower():
                idx = text.lower().find(quote.lower())
                category.quote = text[idx : idx + len(quote)]
            if (
                category.category not in category_options
                or category.quote not in text
                or not compatible_category(category.category, products, category_products)
            ):
                reason = (
                    "category_not_supported"
                    if category.category not in category_options
                    else "untraceable_quote"
                    if category.quote not in text
                    else "incompatible_product"
                )
                raise LanguageUnavailable("unsupported_category_proposal:" + reason)
        assessments = []
        for observation in (result.severity, result.sentiment):
            assessment = None
            if observation and observation.value != "unknown":
                start = text.rfind(observation.quote)
                if start < 0:
                    raise LanguageUnavailable("untraceable_assessment")
                assessment = RuleAssessment(
                    value=observation.value,
                    rule="quoted_language_assessment",
                    method="language_assisted",
                    evidence=[
                        TextEvidence(
                            text=observation.quote, start=start, end=start + len(observation.quote)
                        )
                    ],
                )
            assessments.append(assessment)
        base = (products, facts, category)
        return (*base, *assessments) if include_assessments else base

    if isinstance(client, ProviderChain):
        result = client.generate(
            instruction,
            payload,
            Interpretation,
            validator=validate,
        )
    else:
        result = client.generate(instruction, payload, Interpretation)
    return validate(result)


def validate_interpretation(text, result):
    """Check interpretation before accepting a provider and calculating exact offsets."""
    products, facts = [], []
    for item in [*result.products, *result.facts]:
        start = text.rfind(item.quote)
        if start < 0:
            stripped = item.quote.strip("\"' ")
            start = text.rfind(stripped)
            if start >= 0:
                item.quote = text[start : start + len(stripped)]
            else:
                lower_text = text.lower()
                lower_quote = stripped.lower()
                start = lower_text.rfind(lower_quote)
                if start >= 0:
                    item.quote = text[start : start + len(lower_quote)]
        if start < 0:
            raise LanguageUnavailable("untraceable_observation")
        evidence = dict(text=item.quote, start=start, end=start + len(item.quote))
        if isinstance(item, ServiceMention):
            products.append(ProductObservation(product=item.product, **evidence))
        else:
            if item.name in VALUES and item.value not in VALUES[item.name]:
                raise LanguageUnavailable("invalid_observation_value")
            if item.name == "mobile_services":
                markers = {
                    "calls": r"\b(?:calls?|calling|dial\w*|voice)\b",
                    "texts": r"\b(?:texts?|messages?|SMS|OTP|codes?)\b",
                    "data": r"\b(?:data|cellular browsing|4G|5G)\b",
                }
                mentioned = {
                    name
                    for name, pattern in markers.items()
                    if re.search(pattern, item.quote, re.I)
                }
                # No signal/service does not identify a single failing mobile service.
                if (item.value != "several" and item.value not in mentioned) or (
                    item.value == "several" and len(mentioned) < 2
                ):
                    continue
            if (
                item.name == "equipment_condition"
                and item.value == "damaged"
                and not re.search(
                    r"\b(?:damaged|broken|cracked|burnt|burned|melted|frayed|crushed|snapped)\b",
                    item.quote,
                    re.I,
                )
            ):
                continue
            if (
                item.name == "equipment_condition"
                and item.value == "intact"
                and not re.search(r"\b(?:intact|undamaged|not damaged|not wet)\b", item.quote, re.I)
            ):
                continue
            if item.name == "billing_status" and re.search(
                r"\b(?:whether|maybe|might|not sure|supposed to)\b", item.quote, re.I
            ):
                continue
            if item.name == "billing_status":
                marker = (
                    r"\b(?:pending|authorization only|not yet settled|not cleared)\b"
                    if item.value == "pending"
                    else r"\b(?:settled|completed|cleared)\b"
                )
                if not re.search(marker, item.quote, re.I):
                    continue
                if item.value == "settled" and re.search(
                    r"\b(?:not|isn't|hasn't|never)\b", item.quote, re.I
                ):
                    continue
            if (
                item.name == "wireless_devices"
                and item.value == "all"
                and not re.search(r"\b(?:all|every|both|each)\b", item.quote, re.I)
            ):
                continue
            facts.append(ReportedFact(name=item.name, value=item.value, **evidence))
    for rule_fact in extract_facts(text):
        same = [fact for fact in facts if fact.name == rule_fact.name]
        if not same or rule_fact.start >= max(fact.start for fact in same):
            facts = [fact for fact in facts if fact.name != rule_fact.name] + [rule_fact]
    return products, facts
