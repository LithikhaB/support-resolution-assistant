"""Exercise isolated issue histories, corrections, bounds and advisory follow-up behaviour."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config.settings import Settings
from app.main import app
from app.resolution.conversation import ConversationRequest, resolve_conversation
from app.resolution.service import ResolutionService, get_resolution_service
from app.retrieval.models import BM25Result
from app.understanding.models import CategoryCandidate
from app.understanding.scope import scope_assessment, split_issues
from app.understanding.service import UnderstandingService
from app.understanding.signals import assess_severity


@pytest.fixture
def service():
    """Use real orchestration and rules with deterministic classifier and evidence fixtures."""
    classifier = Mock()
    classifier.artifact.model_dump.return_value = {"version": "conversation-test"}
    classifier.predict.return_value = [
        CategoryCandidate(category="intermittent_broadband", score=0.2),
        CategoryCandidate(category="billing_dispute", score=0.19),
    ]
    understanding = UnderstandingService(
        classifier,
        settings=Settings(
            _env_file=None,
            understanding_routing_path=Path("data/models/absent-test-policy.json"),
        ),
    )
    source = BM25Result(
        chunk_id=1,
        doc_id="test-kb",
        chunk_index=0,
        title="Test procedure",
        doc_type="knowledge_base",
        outcome_status="unknown",
        response=None,
        resolution=None,
        metadata={"authority": "fictional_provider_policy", "is_synthetic": True},
        bm25_score=1,
        bm25_rank=1,
        content="Scope: home_wifi; support category: intermittent_broadband.\nDiagnostic gate: An agent confirms a wireless fault.\nOnly if that finding is established: Restart the router.\nRestriction: Never disclose credentials.",
    )
    retrieval = Mock()
    retrieval.search.return_value = SimpleNamespace(results=[source])
    return ResolutionService(understanding=understanding, retrieval=retrieval)


def run(service, query="My broadband drops. I already restarted the router twice.", turns=None):
    """Run a validated replay and return its first issue for concise assertions."""
    return resolve_conversation(
        ConversationRequest(query=query, turns=turns or []), service
    ).issues[0]


def test_unavailable_ethernet_asks_achievable_alternative(service):
    result = run(
        service,
        turns=[
            {
                "issue_id": 1,
                "message": "Wi-Fi is affected. I don't have any devices with Ethernet, please don't ask me to do that.",
            }
        ],
    ).resolution
    assert any(
        f.name == "wired_connection" and f.value == "unavailable"
        for f in result.analysis.reported_facts
    )
    assert any("Without using Ethernet" in q for q in result.clarification_questions)
    assert not any("connected by Ethernet" in q for q in result.clarification_questions)


@pytest.mark.parametrize(
    "reply",
    [
        "The laptop disconnects over Ethernet at the same time as Wi-Fi.",
        "The wired PC loses internet as well.",
        "I watched the wired PC tonight and it loses internet along with the wireless devices.",
        "Actually Ethernet also drops.",
    ],
)
def test_natural_wired_failure_replaces_previous_working_answer(service, reply):
    """Use the corrected observation and stop repeating the answered comparison."""
    result = run(
        service,
        turns=[
            {"issue_id": 1, "message": "Ethernet works fine on my desktop."},
            {"issue_id": 1, "message": reply},
        ],
    ).resolution
    wired = [f.value for f in result.analysis.reported_facts if f.name == "wired_connection"]
    assert wired and set(wired) == {"failing"}
    assert not any("Ethernet" in q for q in result.clarification_questions)


def test_cannot_try_ethernet_does_not_count_as_failed_test(service):
    """Offer an achievable question when no cable test can be performed."""
    result = run(
        service, turns=[{"issue_id": 1, "message": "Can't try Ethernet, only phones here."}]
    ).resolution
    assert any(
        f.value == "unavailable"
        for f in result.analysis.reported_facts
        if f.name == "wired_connection"
    )
    assert any("Without using Ethernet" in q for q in result.clarification_questions)


def test_conditional_wired_failure_is_not_an_answer(service):
    """A hypothetical outcome must leave the diagnostic question outstanding."""
    result = run(
        service,
        turns=[{"issue_id": 1, "message": "If the wired PC loses internet, what should I do?"}],
    ).resolution
    assert not any(f.name == "wired_connection" for f in result.analysis.reported_facts)
    assert any("connected by Ethernet" in q for q in result.clarification_questions)


def test_billing_with_working_broadband_does_not_ask_for_ethernet(service):
    """Keep a working service comparison from becoming an unrelated repair question."""
    result = run(
        service, query="My bill has an unwanted TV charge. My broadband works fine."
    ).resolution
    assert "broadband" not in {p.product for p in result.analysis.products}
    assert not any("Ethernet" in q for q in result.clarification_questions)


def test_tv_package_charge_is_not_a_tv_picture_fault(service):
    """An invoice item must not trigger a service-availability question."""
    result = run(
        service,
        query="There is a charge for an extra TV package on my bill. The broadband itself is working normally.",
    ).resolution
    assert {p.product for p in result.analysis.products} == {"billing"}
    assert not any("unavailable or intermittent" in q for q in result.clarification_questions)


def test_billing_does_not_hide_genuine_broadband_failure(service):
    """Separate two failing services and retain the connectivity investigation."""
    response = resolve_conversation(
        ConversationRequest(query="My broadband drops. My bill has a duplicate charge."), service
    )
    assert len(response.issues) == 2
    assert any("Ethernet" in q for q in response.issues[0].resolution.clarification_questions)


def test_bill_reply_can_identify_sim_and_request_customer_care(service):
    """Accept the reported UI reply and prioritize its request for human support."""
    result = run(
        service,
        query="I am absolutely furious with this company! My bill has a Rs 50 late fee and every service works fine. Explain why.",
        turns=[
            {
                "issue_id": 1,
                "message": "My SIM card, the bill is still pending. I will pay the bill, I want a CUSTOMER CARE NUMBER.",
            }
        ],
    ).resolution
    assert result.contact_status == "unverified"
    assert result.decision.action == "escalate"
    assert not result.clarification_questions
    assert "human support" in result.customer_plan.title
    assert "No call has been arranged" in result.customer_plan.note


@pytest.mark.parametrize(
    "reply",
    [
        "My SIM card bill, once it clarifies I will settle the bill amount.",
        "The bill is still unpaid.",
        "I will pay the bill after you explain the late fee.",
    ],
)
def test_unpaid_bill_answer_ends_the_payment_status_loop(service, reply):
    """A promise to pay is an unpaid bill, not a pending bank transaction."""
    query = "My bill has a Rs 50 late fee and every service works fine. Explain why."
    initial = run(service, query=query).resolution
    assert any("due date" in question for question in initial.clarification_questions)
    result = run(service, query=query, turns=[{"issue_id": 1, "message": reply}]).resolution
    assert not result.clarification_questions
    assert result.customer_plan.title == "Customer reports an unpaid bill"
    assert "account" in result.customer_plan.note
    assert not any(
        f.name == "billing_status" and f.value == "pending" for f in result.analysis.reported_facts
    )


def test_pending_bank_transaction_is_not_treated_as_an_unpaid_bill(service):
    """Keep the distinction between money sent and a future intention to pay."""
    result = run(service, query="I paid my bill and the payment is pending at the bank.").resolution
    assert result.customer_plan.title != "Your bill is still unpaid"
    assert any(
        f.name == "billing_status" and f.value == "pending" for f in result.analysis.reported_facts
    )


def test_working_services_do_not_close_a_billing_complaint(service):
    """Operational telecom services do not resolve a disputed fee."""
    result = run(service, query="My bill has a late fee. All services are working.").resolution
    assert result.customer_plan.title != "You reported that service is working again"
    assert any("due date" in question for question in result.clarification_questions)


def test_water_in_router_skips_routine_connectivity_questions(service):
    """Water exposure must take priority even when no matching procedure is retrieved."""
    result = run(
        service,
        query="Water got into the broadband router during a leak. It smells burnt and has no lights.",
    ).resolution
    assert "inspection" in result.customer_plan.title
    assert not result.clarification_questions
    assert not result.suggestions


@pytest.mark.parametrize(
    "reply",
    [
        "My SIM has no signal and my bill is pending.",
        "My mobile data is not working and my bill is pending.",
        "My TV has no picture and my bill is pending.",
    ],
)
def test_billing_followup_cannot_add_a_new_technical_fault(reply):
    """Naming the bill's service must not permit an unrelated fault in the same turn."""
    with pytest.raises(ValidationError, match="another service"):
        ConversationRequest(
            query="My bill has a late fee.", turns=[{"issue_id": 1, "message": reply}]
        )


def test_devices_on_wifi_do_not_split_into_mobile_and_tv_issues(service):
    request = ConversationRequest(
        query="Wi-Fi is fine on my laptop and TV. My phone is the only thing that disconnects, usually when I lock the screen."
    )
    response = resolve_conversation(request, service)
    assert len(response.issues) == 1
    assert {p.product for p in response.issues[0].resolution.analysis.products} == {"home_wifi"}


def test_flood_damage_keeps_related_equipment_together(service):
    request = ConversationRequest(
        query="The floodwater got into my modem and the phone wiring. The street has power again, but my equipment is damaged. Can someone arrange a replacement?"
    )
    response = resolve_conversation(request, service)
    assert len(response.issues) == 1
    assert response.issues[0].resolution.decision.target == "field_service"
    assert not response.issues[0].resolution.clarification_questions
    service.retrieval.search.assert_not_called()


def test_landline_gap_is_explicit_instead_of_reasking_service(service):
    result = run(service, query="My landline has been dead for two days.").resolution
    assert "landline" in result.customer_plan.summary
    assert not result.clarification_questions
    assert result.decision.action == "escalate"


def test_followup_advances_question_without_forgetting_restart(service):
    first = run(service)
    assert any("Ethernet" in q for q in first.resolution.clarification_questions)
    second = run(service, turns=[{"issue_id": 1, "message": "Ethernet works."}])
    assert not any("Ethernet" in q for q in second.resolution.clarification_questions)
    assert any("every wireless" in q for q in second.resolution.clarification_questions)
    assert any("restarted" in a for a in second.resolution.acknowledged_actions)
    third = run(
        service,
        turns=[
            {"issue_id": 1, "message": "Ethernet works."},
            {"issue_id": 1, "message": "All wireless devices disconnect."},
        ],
    )
    assert not third.resolution.clarification_questions
    assert third.resolution.suggestions[0].status == "withheld_previously_attempted"
    assert third.resolution.validation.status == "passed"
    for fact in third.resolution.analysis.reported_facts:
        assert third.analysis_text[fact.start : fact.end] == fact.text


def test_latest_turn_corrects_fact_but_same_turn_conflict_stays_visible(service):
    corrected = run(
        service,
        turns=[
            {"issue_id": 1, "message": "Ethernet works."},
            {"issue_id": 1, "message": "Correction: Ethernet drops."},
        ],
    )
    facts = corrected.resolution.analysis.reported_facts
    assert {f.value for f in facts if f.name == "wired_connection"} == {"failing"}
    conflict = run(service, turns=[{"issue_id": 1, "message": "Ethernet works. Ethernet drops."}])
    assert any(
        "conflicting wired connection" in q for q in conflict.resolution.clarification_questions
    )


def test_structured_answers_have_traceable_spans_and_update_impact(service):
    result = run(
        service,
        turns=[
            {
                "issue_id": 1,
                "observations": {
                    "wired_connection": "failing",
                    "impact": "complete_loss",
                },
            }
        ],
    )
    analysis = result.resolution.analysis
    assert analysis.severity.value == "high"
    assert not any("Ethernet" in q for q in result.resolution.clarification_questions)
    for fact in analysis.reported_facts:
        assert result.analysis_text[fact.start : fact.end] == fact.text
    for fact in analysis.severity.evidence:
        assert result.analysis_text[fact.start : fact.end] == fact.text


def test_recovery_withholds_repairs_without_claiming_verified_closure(service):
    result = run(
        service,
        query="My broadband drops.",
        turns=[
            {"issue_id": 1, "observations": {"impact": "working"}},
        ],
    ).resolution
    assert result.analysis.severity.value == "low"
    assert not result.clarification_questions
    assert "before closing" in result.draft
    assert "procedure proposes" not in result.draft
    assert result.agent_review_required
    assert result.validation.status == "passed"


def test_regional_outage_retains_urgency_after_local_recovery_and_area_answer(service):
    result = run(
        service,
        query="Our street lost broadband.",
        turns=[
            {
                "issue_id": 1,
                "observations": {"area": "North street", "started": "8 AM", "impact": "working"},
            },
        ],
    ).resolution
    assert result.decision.priority == "urgent"
    assert result.decision.action == "escalate"
    assert not any("shared outage" in q for q in result.clarification_questions)


def test_multi_issue_replies_do_not_leak_into_other_issue(service):
    result = resolve_conversation(
        ConversationRequest(
            query="My broadband drops. I restarted the router. Also my bill has a duplicate charge.",
            turns=[{"issue_id": 1, "message": "Ethernet works."}],
        ),
        service,
    )
    assert len(result.issues) == 2
    assert result.issues[0].resolution.acknowledged_actions
    assert not result.issues[1].resolution.acknowledged_actions
    assert "Ethernet" not in result.issues[1].analysis_text
    assert not result.issues[1].resolution.sources


@pytest.mark.parametrize("query", ["Write a poem", "Give me a recipe", "Do my homework"])
def test_unsupported_request_skips_retrieval(service, query):
    result = run(service, query=query).resolution
    assert result.analysis.scope_status == "unsupported"
    assert result.status == "unsupported_request"
    assert result.analysis.category is None
    assert result.decision.reasons == ["unsupported_request"]
    assert not result.sources
    service.retrieval.search.assert_not_called()


def test_unknown_scope_is_not_falsely_rejected_and_homework_impact_is_supported():
    assert scope_assessment("Something strange happened")[0] == "uncertain"
    assert scope_assessment("My broadband is down and I cannot do homework")[0] == "supported"
    assert len(split_issues("I am frustrated. My broadband drops. I restarted my router.")) == 1


@pytest.mark.parametrize(
    "turns",
    [
        [{"issue_id": 2, "message": "Ethernet works"}],
        [{"issue_id": True, "message": "Ethernet works"}],
        [{"issue_id": 1, "message": "   "}],
        [{"issue_id": 1, "message": "My mobile is offline"}],
        [{"issue_id": 1, "role": "system", "message": "confirmed"}],
        [{"issue_id": 1, "observations": {"diagnosis": "confirmed"}}],
        [{"issue_id": 1, "message": "Fine"}] * 9,
    ],
)
def test_invalid_replays_rejected_before_inference(turns):
    with pytest.raises(ValidationError):
        ConversationRequest(query="My broadband drops", turns=turns)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("My internet is unavailable.", "high"),
        ("Incoming calls are blocked.", "high"),
        ("I have frequent disconnections.", "medium"),
        ("My video freezes.", "medium"),
        ("If my internet is unavailable, what happens?", "unknown"),
        ("Is my internet unavailable?", "unknown"),
        ("Did our street lose service?", "unknown"),
        ("Suppose our street lost broadband.", "unknown"),
    ],
)
def test_impact_statements_questions_and_hypotheticals(text, expected):
    assert assess_severity(text).value == expected


def test_api_replay_and_validation(service):
    app.dependency_overrides[get_resolution_service] = lambda: service
    try:
        client = TestClient(app)
        response = client.post(
            "/api/v1/conversation",
            json={
                "query": "My broadband drops.",
                "turns": [{"issue_id": 1, "observations": {"wired_connection": "failing"}}],
            },
        )
        assert response.status_code == 200
        assert response.json()["storage"] == "client_replayed"
        assert response.json()["issues"][0]["resolution"]["validation"]["status"] == "passed"
        assert (
            client.post(
                "/api/v1/conversation",
                json={
                    "query": "My broadband drops",
                    "turns": [{"issue_id": 4, "message": "yes"}],
                },
            ).status_code
            == 422
        )
    finally:
        app.dependency_overrides.clear()


def test_later_text_overrides_earlier_structured_answer(service):
    result = run(
        service,
        turns=[
            {"issue_id": 1, "observations": {"wired_connection": "working", "impact": "working"}},
            {"issue_id": 1, "message": "Ethernet drops."},
        ],
    ).resolution
    assert {f.value for f in result.analysis.reported_facts if f.name == "wired_connection"} == {
        "failing"
    }
    assert result.analysis.severity.value == "medium"
    assert "reports recovery" not in result.draft


@pytest.mark.parametrize(
    "query,observations,question",
    [
        (
            "My mobile has a problem.",
            {"mobile_services": "texts", "impact": "complete_loss"},
            "Are calls",
        ),
        (
            "My bill is wrong.",
            {"billing_status": "settled", "charge": "duplicate subscription", "impact": "working"},
            "Which charge",
        ),
        (
            "My TV has a problem.",
            {"tv_symptom": "buffering", "impact": "intermittent"},
            "Is the TV",
        ),
    ],
)
def test_service_specific_answers_stop_repeating_answered_question(
    service, query, observations, question
):
    result = run(
        service, query=query, turns=[{"issue_id": 1, "observations": observations}]
    ).resolution
    assert not any(question in q for q in result.clarification_questions)
    assert result.validation.status == "passed"


def test_structured_observation_for_wrong_service_rejected():
    with pytest.raises(ValidationError):
        ConversationRequest(
            query="My mobile has a problem",
            turns=[
                {"issue_id": 1, "observations": {"billing_status": "settled"}},
            ],
        )


def test_no_cross_request_memory(service):
    run(service, turns=[{"issue_id": 1, "message": "Ethernet works."}])
    result = run(service)
    assert any("Ethernet" in q for q in result.resolution.clarification_questions)


def test_multi_issue_can_separate_unsupported_request(service):
    result = resolve_conversation(
        ConversationRequest(query="My broadband drops. Also write a poem."), service
    )
    assert len(result.issues) == 2
    assert result.issues[0].resolution.analysis.scope_status == "supported"
    assert result.issues[1].resolution.analysis.scope_status == "unsupported"
    assert service.retrieval.search.call_count == 2


def test_complete_loss_takes_precedence_over_degradation():
    assert assess_severity("Frequent disconnections. Now no internet.").value == "high"


def test_repeated_reply_text_keeps_offsets_in_its_own_turn(service):
    result = run(
        service,
        query="My broadband drops. Ethernet works.",
        turns=[
            {"issue_id": 1, "message": "Ethernet drops."},
            {"issue_id": 1, "message": "Ethernet works."},
        ],
    )
    facts = [f for f in result.resolution.analysis.reported_facts if f.name == "wired_connection"]
    assert len(facts) == 1 and facts[0].value == "working"
    assert facts[0].start > len(result.complaint)


@pytest.mark.parametrize(
    "text",
    [
        "Do not write a poem. My broadband is down.",
        "My broadband is slow when checking the weather forecast.",
    ],
)
def test_unrelated_topic_mention_does_not_reject_a_service_complaint(text):
    assert scope_assessment(text)[0] == "supported"


@pytest.mark.parametrize(
    "text",
    [
        "My router was soaked during a flood. I need an immediate replacement.",
        "In the floods, the ethernet line was broken, now everything is restored in my area, but due to flood waters, all the modem, cables, landlines everything is damaged. I request an immediate replacement.",
        "My modem is damaged. I need a replacement.",
    ],
)
def test_physical_damage_withholds_unrelated_repairs_and_retrieval(service, text):
    """Report damage without treating regional recovery as safe equipment or a booked repair."""
    result = run(service, query=text).resolution
    service.retrieval.search.assert_not_called()
    assert result.validation.status == "passed"
    assert result.decision.target == "field_service"
    assert not result.suggestions
    assert not result.clarification_questions
    assert "inspection" in result.customer_plan.title
    assert "not been booked" in result.customer_plan.note
    assert "channel" not in result.draft


@pytest.mark.parametrize(
    "text",
    [
        "My router is not damaged. My broadband drops.",
        "If my router is damaged, would I need a replacement? My broadband drops.",
        "My router might be wet. My broadband drops.",
        "My router works but my internet connection is broken.",
    ],
)
def test_negated_or_hypothetical_damage_does_not_confirm_damage(service, text):
    """A mention of a damaged router is not automatically a reported equipment fault."""
    result = run(service, query=text).resolution
    assert result.decision.target != "field_service"
    assert [call.args[0].filters.doc_type for call in service.retrieval.search.call_args_list] == [
        "knowledge_base",
        "resolved_ticket",
    ]


def test_current_correction_can_clear_old_damage_but_connectivity_answer_cannot(service):
    """Only an explicit equipment correction clears the earlier equipment report."""
    query = "My router is damaged by flood water. My broadband drops."
    retained = run(
        service,
        query=query,
        turns=[
            {"issue_id": 1, "observations": {"wired_connection": "working", "impact": "working"}}
        ],
    ).resolution
    assert retained.decision.target == "field_service"
    assert "wet equipment" in retained.customer_plan.steps[0]
    corrected = run(
        service,
        query=query,
        turns=[
            {"issue_id": 1, "message": "Correction: my router is undamaged. My broadband drops."}
        ],
    ).resolution
    assert corrected.decision.target != "field_service"


def test_human_contact_request_does_not_start_a_provider_lookup(service):
    """A call preference prompts desk follow-up without invented numbers or provider questions."""
    result = run(service, query="My broadband drops. Give me a toll-free number.").resolution
    assert result.contact_status == "unverified"
    assert not any("provider" in q for q in result.clarification_questions)
    assert "telephone directory" in result.draft


def test_customer_plan_tampering_is_rejected(service):
    """Customer-facing steps receive the same integrity protection as the internal draft."""
    from app.resolution.validation import validate_citations

    result = run(service, query="My modem is damaged.").resolution
    result.customer_plan.steps.append("Replacement booked for tomorrow, free of charge.")
    checked = validate_citations(result, [], result.analysis)
    assert "customer_plan_changed" in checked.issues
