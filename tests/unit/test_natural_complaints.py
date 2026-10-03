"""Regressions for user-reported failures, with contrasting and negated observations."""

from types import SimpleNamespace

import pytest

from app.resolution.evidence import supported_scopes
from app.understanding.context import clarification_questions, extract_facts
from app.understanding.language import Interpretation, validate_interpretation
from app.understanding.scope import split_issues
from app.understanding.signals import assess_severity, extract_actions, extract_products


def questions(text):
    """Run the offline clarification path without loading models or calling providers."""
    return clarification_questions(
        accepted=False,
        products=extract_products(text),
        severity=assess_severity(text),
        facts=extract_facts(text),
        requests=[],
    )


def test_cabled_failure_answers_the_wired_question():
    text = "My internet cuts out. I plugged my laptop into the router with a cable and it still drops at the same moments the Wi-Fi does."
    facts = extract_facts(text)
    assert ("wired_connection", "failing") in {(f.name, f.value) for f in facts}
    assert not any("Ethernet" in q for q in questions(text))
    assert all(text[f.start : f.end] == f.text for f in facts)


@pytest.mark.parametrize(
    "text",
    [
        "If I connect with a cable and it drops, should I call?",
        "With a cable it never drops.",
        "With a cable it works, but Wi-Fi drops.",
        "My TV no longer freezes.",
        "The live channels are not freezing.",
    ],
)
def test_negation_and_other_service_failure_do_not_invent_a_fault(text):
    assert not any(f.name in {"wired_connection", "tv_symptom"} for f in extract_facts(text))


def test_roaming_data_and_bank_codes_do_not_trigger_generic_service_questions():
    for text in [
        "My travel pack is active but I have no data at all, though I can make calls.",
        "I can't get my bank's one-time passwords on my phone. Only the bank ones are missing.",
    ]:
        assert not any("Are calls" in q for q in questions(text))


def test_uncertain_payment_does_not_become_settled_or_pending():
    text = "I see two charges on this month's bill. Not sure whether one is a hold."
    facts = extract_facts(text)
    assert not any(f.name == "billing_status" for f in facts)
    assert len(questions(text)) == 1 and "pending or settled" in questions(text)[0]


def test_live_tv_and_working_comparison_form_one_issue_with_tv_evidence():
    text = "The live channels freeze every few seconds. Netflix on my phone is fine on the same Wi-Fi. The set-top box is in the bedroom, far from the router."
    assert split_issues(text) == [text]
    analysis = SimpleNamespace(products=extract_products(text), reported_facts=extract_facts(text))
    assert supported_scopes(analysis) == {"iptv"}
    assert not any("During a drop" in q for q in questions(text))
    assert len(split_issues("The live channels freeze. My mobile data also fails.")) == 2


def test_unplugging_is_retained_as_attempt_not_a_new_repair():
    actions = extract_actions(
        "The router light stays green. I unplugged it for ten minutes, twice."
    )
    assert any(a.action == "restart_device" and a.status == "attempted" for a in actions)


def test_green_light_does_not_prove_intact_equipment():
    text = "The router light stays green."
    _, facts = validate_interpretation(
        text,
        Interpretation(
            products=[], facts=[{"name": "equipment_condition", "value": "intact", "quote": text}]
        ),
    )
    assert not facts


@pytest.mark.parametrize(
    "service,category,expected",
    [
        ("texts", None, {"mobile_sms"}),
        ("calls", None, {"mobile_voice"}),
        ("data", None, {"mobile_data"}),
        ("data", "roaming", {"mobile_roaming"}),
    ],
)
def test_mobile_evidence_follows_failing_service(service, category, expected):
    analysis = SimpleNamespace(
        category=category,
        products=[SimpleNamespace(product="mobile")],
        reported_facts=[SimpleNamespace(name="mobile_services", value=service)],
        severity=SimpleNamespace(rule="test"),
    )
    assert supported_scopes(analysis) == expected


@pytest.mark.parametrize(
    "quote,value",
    [
        ("two charges on my bank statement", "pending"),
        ("Not sure whether this is pending", "pending"),
        ("The payment has not settled", "settled"),
    ],
)
def test_language_cannot_turn_charge_count_or_uncertainty_into_payment_status(quote, value):
    _, facts = validate_interpretation(
        quote,
        Interpretation(
            products=[], facts=[{"name": "billing_status", "value": value, "quote": quote}]
        ),
    )
    assert not any(f.name == "billing_status" for f in facts)
