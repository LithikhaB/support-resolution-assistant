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
    result = understanding.analyze(
        AnalyzeRequest(
            query="My number transfer was rejected because the details do not match the old provider's account."
        )
    )
    assert result.category == "number_porting"
    assert result.scope_status == "supported"
    assert result.language_method == "rules_fallback"
    assert questions_for(result) == []


def test_fallback_separates_check_action_and_restriction():
    from app.resolution.customer import procedure_note, procedure_steps
    from app.resolution.models import ConditionalSuggestion

    result = Mock(
        acknowledged_actions=[],
        analysis=Mock(actions=[], reported_facts=[]),
        sources=[],
        historical_cases=[],
        suggestions=[
            ConditionalSuggestion(
                citation_id="S1",
                required_finding="Ledger confirms duplicate settled payments.",
                proposed_action="Authorized billing staff reconcile the duplicate payment.",
                restriction="Do not promise a refund deadline.",
                status="requires_agent_confirmation",
            )
        ],
    )
    steps = procedure_steps(result)
    assert len(steps) == 2 and all("[S1]" in step for step in steps)
    assert steps[0].startswith("Check with authorized")
    assert steps[1].startswith("Only if support confirms")
    assert "refund deadline" in procedure_note(result)


def test_language_selection_only_accepts_retrieved_applicable_ids_and_redacts_payload():
    from app.llm.client import LanguageUnavailable
    from app.resolution.selection import ProcedureChoice, select_relevant_procedure

    rows = [
        BM25Result(
            chunk_id=i,
            doc_id=f"kb{i}",
            chunk_index=0,
            title="Billing investigation",
            doc_type="knowledge_base",
            response=None,
            resolution=None,
            outcome_status="reference",
            metadata={"authority": "fictional_provider_policy", "is_synthetic": True},
            bm25_score=1,
            bm25_rank=i,
            content=f"Scope: billing; support category: billing_dispute.\nDiagnostic gate: Ledger check {i}.\nOnly if that finding is established: Authorized billing staff reconcile the ledger.\nRestriction: Do not promise a refund deadline.",
        )
        for i in (1, 2)
    ]
    observed = service().analyze(AnalyzeRequest(query="My bill has two payments."))
    client = Mock()
    client.generate.return_value = ProcedureChoice(doc_id="kb2")
    assert select_relevant_procedure(
        rows, "My bill has two payments; email me at customer@example.com.", observed, client
    ) == [rows[1]]
    assert "customer@example.com" not in str(client.generate.call_args.args[1])
    client.generate.return_value = ProcedureChoice(doc_id="invented")
    with pytest.raises(LanguageUnavailable, match="invalid_selected_procedure"):
        select_relevant_procedure(rows, "My bill has two payments.", observed, client)
    client.generate.return_value = ProcedureChoice(doc_id=None)
    assert select_relevant_procedure(rows, "My bill has two payments.", observed, client) == []


def test_no_mobile_service_does_not_become_an_invented_calls_only_scope():
    from app.resolution.evidence import supported_scopes
    from app.understanding.language import validate_interpretation

    text = "My phone has no service after I manually selected a network."
    products, facts = validate_interpretation(
        text,
        Interpretation(
            products=[{"product": "mobile", "quote": text}],
            facts=[{"name": "mobile_services", "value": "calls", "quote": text}],
        ),
    )
    observed = service().analyze(AnalyzeRequest(query=text))
    observed.products, observed.reported_facts = products, facts
    assert not any(f.name == "mobile_services" for f in facts)
    assert "mobile" in supported_scopes(observed)


def test_overheating_is_not_inferred_physical_damage():
    from app.understanding.language import validate_interpretation

    text = "The router gets hot in a closed cabinet and shuts down."
    _, facts = validate_interpretation(
        text,
        Interpretation(
            products=[{"product": "router", "quote": "router"}],
            facts=[{"name": "equipment_condition", "value": "damaged", "quote": text}],
        ),
    )
    assert not any(f.name == "equipment_condition" for f in facts)


def test_porting_scope_keeps_transfer_procedures_when_only_calls_fail():
    from app.resolution.evidence import supported_scopes
    from app.understanding.models import ReportedFact

    observed = service().analyze(AnalyzeRequest(query="After number porting, incoming calls fail."))
    observed.category = "number_porting"
    observed.reported_facts = [
        ReportedFact(name="mobile_services", value="calls", text="calls", start=0, end=5)
    ]
    assert "mobile" in supported_scopes(observed)


def test_freeform_reviewed_checks_are_available_to_drafting():
    from app.resolution.language import reported_checks

    text = "My bill is higher. Already checked: Compared invoice dates with the plan-change date. Could you check this when you have the details?"
    assert reported_checks(text) == ["Compared invoice dates with the plan-change date."]
