"""Guard against local troubleshooting for shared outages and unsupported routing."""

from unittest.mock import Mock

import numpy as np
import pytest

from app.config.settings import Settings
from app.understanding.context import clarification_questions
from app.understanding.models import AnalyzeRequest, CategoryCandidate
from app.understanding.routing import compatible_category, load_policy
from app.understanding.service import UnderstandingService
from app.understanding.signals import assess_severity, extract_products
from scripts.calibrate_routing import select_thresholds


@pytest.mark.parametrize(
    "text",
    [
        "All the flats on my street lost fibre service together.",
        "The entire area has no internet.",
        "Several neighbours lost broadband at the same time.",
    ],
)
def test_shared_outage_is_critical_and_does_not_ask_wifi_question(text):
    severity = assess_severity(text)
    assert severity.value == "critical"
    questions = clarification_questions(
        accepted=False, products=extract_products(text), severity=severity, facts=[], requests=[]
    )
    assert any("shared outage" in q for q in questions)
    assert not any("Ethernet" in q or "Wi-Fi" in q for q in questions)


@pytest.mark.parametrize(
    "text",
    [
        "Our street has not lost broadband.",
        "If our street lost broadband, whom should I call?",
        "Maybe our street lost broadband.",
        "My router is offline. Our street has a grocery shop.",
    ],
)
def test_negated_hypothetical_or_unrelated_area_does_not_escalate(text):
    assert assess_severity(text).value != "critical"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("My line remains suspended.", "high"),
        ("Calls fail immediately.", "high"),
        ("The router gets hot and shuts down.", "high"),
        ("The TV shows no picture.", "medium"),
        ("If calls fail immediately, should I reboot?", "unknown"),
    ],
)
def test_explicit_impact_and_hypothetical_impact(text, expected):
    assert assess_severity(text).value == expected


def test_reported_shared_broadband_outage_overrides_conflicting_model_with_provenance():
    classifier = Mock()
    classifier.artifact.model_dump.return_value = {"version": "test"}
    classifier.predict.return_value = [
        CategoryCandidate(category="mobile_coverage", score=0.9),
        CategoryCandidate(category="broadband_outage", score=0.05),
    ]
    result = UnderstandingService(classifier, settings=Settings(_env_file=None)).analyze(
        AnalyzeRequest(query="Our street and neighbouring blocks all lost broadband at once.")
    )
    assert result.category == "broadband_outage" and result.category_basis == "explicit_report"
    assert result.category_evidence
    assert result.candidates[0].category == "mobile_coverage"


def test_model_category_requires_compatible_explicit_service():
    assert not compatible_category("mobile_coverage", extract_products("My broadband drops."))
    assert not compatible_category("billing_dispute", extract_products("Bake a cake."))
    assert compatible_category("sms_otp", extract_products("My bank login text is missing."))
    assert "mobile" not in {p.product for p in extract_products("Support phone number please")}


def test_missing_or_stale_routing_profile_preserves_fixed_defaults(tmp_path):
    settings = Settings(_env_file=None, understanding_routing_path=tmp_path / "routing.json")
    assert load_policy(settings, "model") is None
    settings.understanding_routing_path.write_text("{}")
    assert load_policy(settings, "model") is None


def test_routing_selection_requires_accuracy_and_multiple_families():
    rows = [
        {
            "query": "Broadband drops",
            "scenario_family": str(i % 5),
            "labels": {"intent": "broadband_outage"},
        }
        for i in range(20)
    ]
    scores = np.array([[0.8, 0.1, 0.1]] * 20)
    classes = ["broadband_outage", "mobile_coverage", "billing_dispute"]
    selected = select_thresholds(rows, scores, classes)
    assert selected["accepted"] == 20 and selected["accepted_families"] == 5
    for row in rows:
        row["scenario_family"] = "one"
    assert select_thresholds(rows, scores, classes) is None
