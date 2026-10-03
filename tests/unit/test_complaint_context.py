"""Regression cases for reported conditions and context-sensitive clarification."""

from unittest.mock import Mock

import pytest

from app.config.settings import Settings
from app.understanding.context import clarification_questions, extract_facts, extract_requests
from app.understanding.models import AnalyzeRequest, CategoryCandidate, ReportedFact
from app.understanding.service import UnderstandingService
from app.understanding.signals import assess_severity, extract_products


@pytest.mark.parametrize(
    "text",
    [
        "Is the cable intact?",
        "If the cable is intact, restart.",
        "The cable is not intact.",
        "The cable is broken but my router is fine.",
        "Maybe the cable is intact.",
    ],
)
def test_does_not_invent_intact_cable(text):

    assert not any(f.value == "intact" for f in extract_facts(text))


def test_facts_preserve_unicode_offsets_and_conflicting_reports():

    text = "Café: cable in my home is intact. The cable is damaged. Ethernet works."

    facts = extract_facts(text)

    assert [(f.name, f.value) for f in facts] == [
        ("cable_condition", "intact"),
        ("cable_condition", "damaged"),
        ("equipment_condition", "damaged"),
        ("wired_connection", "working"),
    ]

    assert all(text[f.start : f.end] == f.text for f in facts)


def test_negated_contact_request_is_not_requested():

    assert extract_requests("I don't need a helpline number.") == []


def test_known_payment_status_is_not_requested_again_without_charge_description():
    text = "Both transactions are settled."
    fact = ReportedFact(name="billing_status", value="settled", text=text, start=0, end=len(text))
    questions = clarification_questions(
        accepted=False,
        products=extract_products("My bill has a payment problem"),
        severity=assess_severity(text),
        facts=[fact],
        requests=[],
    )
    assert len(questions) == 1 and "Which bill or charge" in questions[0]
    assert "pending or settled" not in questions[0]


def test_settled_duplicate_description_does_not_repeat_known_billing_details():
    text = "My statement has two settled charges for the same monthly bill."
    facts = extract_facts(text)
    questions = clarification_questions(
        accepted=False,
        products=extract_products(text),
        severity=assess_severity(text),
        facts=facts,
        requests=[],
    )
    assert {fact.name for fact in facts} >= {"billing_status", "charge"}
    assert questions == []


def test_user_complaint_retains_context_without_fabricating_category():

    classifier = Mock()

    classifier.artifact.model_dump.return_value = {"version": "test"}

    classifier.predict.return_value = [
        CategoryCandidate(category="intermittent_broadband", score=0.162),
        CategoryCandidate(category="broadband_outage", score=0.160),
    ]

    query = "My broadband drops every evening. I already restarted the router twice. Cable in my home is intact, what to do now? is there any helpline number to contact?"

    result = UnderstandingService(classifier, settings=Settings(_env_file=None)).analyze(
        AnalyzeRequest(query=query)
    )

    assert result.category is None

    assert result.actions[0].status == "attempted"

    assert {r.kind for r in result.requests} == {"contact_support", "next_steps"}

    assert any(f.name == "cable_condition" and f.value == "intact" for f in result.reported_facts)

    assert any("Ethernet" in q for q in result.clarification_questions)

    assert not any("provider" in q for q in result.clarification_questions)

    assert not any("Which service" in q for q in result.clarification_questions)


def test_known_wired_result_changes_follow_up():

    from app.understanding.context import clarification_questions
    from app.understanding.signals import assess_severity, extract_products

    text = "Wi-Fi drops. Ethernet works."

    questions = clarification_questions(
        accepted=False,
        products=extract_products(text),
        severity=assess_severity(text),
        facts=extract_facts(text),
        requests=[],
    )

    assert len(questions) == 1

    assert "every wireless device" in questions[0]
