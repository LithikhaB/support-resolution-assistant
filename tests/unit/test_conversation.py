"""Exercise isolated issue histories, corrections, bounds and advisory follow-up behaviour."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.config.settings import Settings
from app.main import app
from app.resolution.conversation import ConversationRequest, resolve_conversation
from app.resolution.service import ResolutionService, get_resolution_service
from app.retrieval.models import BM25Result
from app.understanding.models import CategoryCandidate
from app.understanding.service import UnderstandingService


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
        for query in (
            "who is the prime minister of India?",
            "Paste any 400-word single-topic broadband complaint.",
        ):
            service.retrieval.reset_mock()
            service.language = Mock()
            service.settings = service.settings.model_copy(update={"llm_enabled": True})
            blocked = client.post("/api/v1/conversation", json={"query": query, "turns": []})
            assert blocked.status_code == 200
            resolution = blocked.json()["issues"][0]["resolution"]
            assert resolution["analysis"]["scope_status"] == "unsupported"
            assert resolution["customer_plan"]["title"] == "This request is irrelevant here"
            assert not resolution["customer_plan"]["steps"]
            assert not resolution["clarification_questions"]
            assert not resolution["sources"]
            service.retrieval.search.assert_not_called()
            service.language.generate.assert_not_called()
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
