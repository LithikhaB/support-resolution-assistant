"""Offline release checkpoints for the user-visible failures in the project review."""

from unittest.mock import Mock

import pytest

from app.config.settings import Settings
from app.llm.providers import ProviderChain
from app.resolution.drafting import draft_resolution
from app.resolution.language import DraftIntroduction, FaithfulnessReview, add_language_draft
from app.resolution.validation import finalize_resolution
from app.retrieval.models import BM25Result
from app.understanding.classifier import ClassifierArtifact
from app.understanding.language import Interpretation
from app.understanding.models import AnalyzeRequest, CategoryCandidate
from app.understanding.service import UnderstandingService
from app.understanding.signals import assess_severity


def service(client=None):
    classifier = Mock()
    classifier.artifact = ClassifierArtifact(
        feature_type="tfidf",
        classes=["broadband_outage", "wifi_connectivity", "number_porting"],
        coefficients=[[1], [0], [0]],
        intercept=[0, 0, 0],
        vocabulary={"internet": 0},
        idf=[1],
        train_sha256="train",
        dev_sha256="dev",
        training_families=["a"],
        development_families=["b"],
        sklearn_version="test",
        regularization_c=1,
    )
    classifier.predict.return_value = [
        CategoryCandidate(category="broadband_outage", score=0.2),
        CategoryCandidate(category="wifi_connectivity", score=0.19),
    ]
    return UnderstandingService(
        classifier,
        settings=Settings(_env_file=None, llm_enabled=client is not None),
        language=client,
    )


def test_semantic_category_can_recover_from_wrong_local_shortlist():
    text = "My number transfer was rejected by the old provider."
    client = Mock()
    client.generate.return_value = Interpretation(
        products=[{"product": "mobile", "quote": "number transfer"}],
        facts=[],
        category={"category": "number_porting", "quote": text},
    )
    result = service(client).analyze(AnalyzeRequest(query=text))
    assert result.category == "number_porting"
    assert result.category_basis == "language_assisted"
    assert "number_porting" in client.generate.call_args.args[1]["category_options"]
    assert client.generate.call_args.args[1]["examples"]["number_porting"]


def test_quoted_impact_sets_severity_when_local_wording_is_unfamiliar():
    text = "My connection behaves like a slideshow during meetings."
    client = Mock()
    client.generate.return_value = Interpretation(
        products=[{"product": "broadband", "quote": "connection"}],
        facts=[{"name": "impact", "value": "degraded", "quote": text}],
    )
    result = service(client).analyze(AnalyzeRequest(query=text))
    assert result.severity.value == "medium"
    assert result.severity.evidence[0].text == text


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Every shop on our lane lost broadband and power at 9am.", "critical"),
        ("Neighbours and I went offline at the same moment.", "critical"),
        ("My internet is down.", "high"),
        ("My Wi-Fi keeps disconnecting.", "medium"),
        ("My bill is higher after changing plans.", "low"),
        ("I am furious.", "unknown"),
        ("If my internet is down, should I call support?", "unknown"),
    ],
)
def test_severity_checkpoint(text, expected):
    assert assess_severity(text).value == expected


def test_clarification_always_has_help_and_does_not_repeat_answered_ethernet_check():
    understanding = service()
    text = "My broadband drops. A device connected by Ethernet also loses internet."
    analysis = understanding.analyze(AnalyzeRequest(query=text))
    assert not any("Ethernet" in q for q in analysis.clarification_questions)
    response = draft_resolution(analysis, [])
    assert response.customer_plan.steps


def test_shared_outage_never_asks_customer_to_test_ethernet():
    analysis = service().analyze(
        AnalyzeRequest(query="Every shop on our lane lost broadband and power at 9am.")
    )
    result = draft_resolution(analysis, [])
    assert result.decision.action == "escalate"
    assert result.decision.target == "network_operations"
    assert not any("Ethernet" in q for q in result.clarification_questions)
    assert "incident" in result.customer_plan.steps[0]


def test_llm_can_produce_a_cited_troubleshooting_step_in_the_current_ui():
    text = "My Wi-Fi drops. Ethernet works. All wireless devices disconnect."
    analysis = service().analyze(AnalyzeRequest(query=text))
    source = BM25Result(
        chunk_id=1,
        doc_id="wifi_kb",
        chunk_index=0,
        title="Coverage investigation",
        doc_type="knowledge_base",
        response=None,
        resolution=None,
        outcome_status="reference",
        metadata={"authority": "fictional_provider_policy", "is_synthetic": True},
        bm25_score=1,
        bm25_rank=1,
        content="Scope: home_wifi; support category: wifi_connectivity.\nDiagnostic gate: Signal survey finds weak coverage in the affected room.\nOnly if that finding is established: Reposition the router and repeat the coverage test.\nRestriction: Do not disable Wi-Fi security.",
    )
    response = finalize_resolution(draft_resolution(analysis, [source]), [source], analysis)
    provider = Mock()
    provider.name, provider.model = "groq", "test-model"
    provider.generate.side_effect = [
        DraftIntroduction(
            summary="Check the affected room's coverage before changing placement.",
            history_citations=[],
            procedure_citations=["S1"],
            steps=[
                {
                    "instruction": "Ask support to survey the Wi-Fi signal in the affected room.",
                    "citation_id": "S1",
                }
            ],
        ),
        FaithfulnessReview(supported=True, issues=[]),
    ]
    settings = Settings(_env_file=None, llm_enabled=True)
    result = add_language_draft(
        response, text, settings, ProviderChain(settings, providers=[provider])
    )
    assert result.language_status == "generated_for_review"
    assert result.language_plan.steps == [
        "Ask support to survey the Wi-Fi signal in the affected room. [S1]"
    ]
    assert result.faithfulness_status == "model_checked"
    assert result.validation.status == "passed"


def test_language_assessments_fill_unknown_severity_and_sentiment():
    text = "My connection behaves like a slideshow during meetings. This is driving me up the wall."
    client = Mock()
    client.generate.return_value = Interpretation(
        products=[{"product": "broadband", "quote": "connection"}],
        facts=[],
        severity={
            "value": "medium",
            "quote": "My connection behaves like a slideshow during meetings.",
        },
        sentiment={"value": "frustrated", "quote": "This is driving me up the wall."},
    )
    result = service(client).analyze(AnalyzeRequest(query=text))
    assert result.severity.value == "medium" and result.severity.method == "language_assisted"
    assert result.sentiment.value == "frustrated" and result.sentiment.method == "language_assisted"


def test_supplied_credentials_are_removed_from_complaint_and_followup():
    from app.resolution.conversation import CustomerTurn

    request = AnalyzeRequest(query="My Wi-Fi password is ExampleSecret123, please fix it.")
    turn = CustomerTurn(issue_id=1, message="My OTP is 123456.")
    assert "ExampleSecret123" not in request.model_dump_json()
    assert "123456" not in turn.model_dump_json()


def test_number_transfer_remains_actionable_during_provider_outage():
    from app.llm.client import LanguageUnavailable
    from app.resolution.policy import questions_for

    client = Mock()
    client.generate.side_effect = LanguageUnavailable("providers_unavailable")
    understanding = service(client)
    understanding.classifier.predict.return_value = [
        CategoryCandidate(category="number_porting", score=0.85),
        CategoryCandidate(category="sim_esim_activation", score=0.02),
    ]
    result = understanding.analyze(AnalyzeRequest(
        query="My number transfer was rejected because the details do not match the old provider's account."
    ))
    assert result.category == "number_porting"
    assert result.scope_status == "supported"
    assert result.language_method == "rules_fallback"
    assert questions_for(result) == []
