"""Challenge persistence, review provenance, stale writes and safe browser delivery."""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app
from app.resolution.conversation import ConversationRequest, ConversationResponse, IssueResponse
from app.resolution.drafting import draft_resolution
from app.resolution.evidence import supported_scopes
from app.resolution.validation import finalize_resolution
from app.understanding.context import clarification_questions, extract_facts
from app.understanding.models import AnalysisResponse
from app.understanding.signals import assess_severity, extract_products
from app.workflow.models import FollowupRequest, ReviewRequest
from app.workflow.service import WorkflowService, get_workflow_service
from app.workflow.store import CaseStore, RevisionConflict


def generated(request):
    """Return a valid generated response without inference or PostgreSQL dependencies."""
    analysis = AnalysisResponse(
        category=None,
        category_status="uncertain",
        candidates=[],
        products=[],
        severity={"value": "unknown", "rule": "test"},
        sentiment={"value": "unknown", "rule": "test"},
        actions=[],
        needs_clarification=True,
        clarification_questions=["Which service?"],
        model_version="test",
        elapsed_ms=0,
    )
    response = finalize_resolution(draft_resolution(analysis, []), [], analysis)
    return ConversationResponse(
        issues=[
            IssueResponse(
                issue_id=1,
                complaint=request.query,
                analysis_text=request.query,
                resolution=response,
            )
        ]
    )


@pytest.fixture
def workflow(tmp_path):
    service = WorkflowService(CaseStore(tmp_path / "cases.sqlite3"))
    service.generate = Mock(side_effect=generated)
    return service


def create(workflow):
    """Start a case using the same request contract as the API."""
    return workflow.create(ConversationRequest(query="My broadband drops."))


def test_saved_case_survives_a_new_store_instance(workflow):
    case = create(workflow)
    reloaded = CaseStore(workflow.store.path).get(case.id)
    assert reloaded == case
    assert workflow.store.recent()[0]["id"] == str(case.id)


def test_agent_edit_does_not_inherit_original_citation_validation(workflow):
    case = create(workflow)
    original = case.response.model_dump_json()
    result = workflow.review(
        case.id,
        ReviewRequest(
            expected_revision=1,
            issue_id=1,
            action="edit",
            edited_draft="Agent-authored replacement",
            notes="Clarified the next check",
        ),
    )
    event = result.reviews[-1]
    assert result.response.model_dump_json() == original
    assert event.text_validation == "agent_edit_not_validated"
    assert event.reviewed_text == "Agent-authored replacement"
    assert event.original_resolution == case.response.issues[0].resolution


def test_accept_records_exact_text_and_reported_outcome(workflow):
    case = create(workflow)
    result = workflow.review(
        case.id,
        ReviewRequest(
            expected_revision=1,
            issue_id=1,
            action="accept",
            outcome="resolved",
            outcome_notes="Customer reported successful retest",
        ),
    )
    assert result.reviews[-1].reviewed_text == case.response.issues[0].resolution.draft
    assert result.reviews[-1].review.outcome == "resolved"
    assert not result.response.issues[0].resolution.decision.handoff_created


def test_rejection_is_audited_without_publishable_review_text(workflow):
    case = create(workflow)
    result = workflow.review(
        case.id,
        ReviewRequest(expected_revision=1, issue_id=1, action="reject", notes="Not applicable"),
    )
    assert result.reviews[-1].reviewed_text is None
    assert result.reviews[-1].text_validation == "rejected"


def test_stale_update_rejected_without_inference(workflow):
    case = create(workflow)
    workflow.review(case.id, ReviewRequest(expected_revision=1, issue_id=1, action="accept"))
    with pytest.raises(RevisionConflict):
        workflow.followup(
            case.id,
            FollowupRequest(
                expected_revision=1, turn={"issue_id": 1, "message": "Ethernet works."}
            ),
        )
    assert workflow.generate.call_count == 1
    assert workflow.store.get(case.id).revision == 2


def test_followup_preserves_old_review_but_requires_fresh_review(workflow):
    case = create(workflow)
    workflow.review(case.id, ReviewRequest(expected_revision=1, issue_id=1, action="accept"))
    updated = workflow.followup(
        case.id,
        FollowupRequest(expected_revision=2, turn={"issue_id": 1, "message": "Ethernet works."}),
    )
    assert updated.generation == 2 and updated.revision == 3
    assert updated.reviews[0].generation == 1
    assert updated.request.turns[0].message == "Ethernet works."
    assert workflow.handoff(case.id, 1)["agent_review"] is None


def test_failed_generation_does_not_modify_saved_history(workflow):
    case = create(workflow)
    workflow.generate.side_effect = RuntimeError("unavailable")
    with pytest.raises(RuntimeError):
        workflow.followup(
            case.id,
            FollowupRequest(
                expected_revision=1, turn={"issue_id": 1, "message": "Ethernet works."}
            ),
        )
    assert workflow.store.get(case.id) == case


def test_simultaneous_writers_cannot_overwrite_each_other(workflow):
    case = create(workflow)
    snapshots = [workflow.store.get(case.id) for _ in range(2)]
    for snapshot in snapshots:
        snapshot.revision = 2

    def save(snapshot):
        """Return an observable conflict outcome from an independent SQLite connection."""
        try:
            workflow.store.save(snapshot, 1)
            return "saved"
        except RevisionConflict:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(save, snapshots)) == ["conflict", "saved"]


@pytest.mark.parametrize(
    "values",
    [
        {"action": "edit"},
        {"action": "accept", "edited_draft": "changed"},
        {"action": "reject", "notes": "   "},
        {"action": "accept", "outcome": "resolved"},
        {"action": "edit", "edited_draft": " ", "notes": "reason"},
    ],
)
def test_review_requires_meaningful_details(values):
    with pytest.raises(ValidationError):
        ReviewRequest(expected_revision=1, issue_id=1, **values)


def test_failed_citation_check_cannot_be_accepted(workflow):
    case = create(workflow)
    case.response.issues[0].resolution.validation.status = "failed"
    case.revision = 2
    workflow.store.save(case, 1)
    with pytest.raises(ValueError, match="failed validation"):
        workflow.review(case.id, ReviewRequest(expected_revision=2, issue_id=1, action="accept"))


def test_handoff_is_local_and_contains_review_provenance(workflow):
    case = create(workflow)
    workflow.review(
        case.id,
        ReviewRequest(
            expected_revision=1,
            issue_id=1,
            action="edit",
            edited_draft="Follow up tomorrow",
            notes="Customer availability",
        ),
    )
    handoff = workflow.handoff(case.id, 1)
    assert handoff["delivery"] == "local_draft_only"
    assert handoff["handoff_created"] is False
    assert handoff["agent_review"].text_validation == "agent_edit_not_validated"


def test_api_case_lifecycle_and_stale_errors(workflow):
    app.dependency_overrides[get_workflow_service] = lambda: workflow
    try:
        client = TestClient(app)
        response = client.post("/api/v1/cases", json={"query": "Broadband drops"})
        assert response.status_code == 201
        case_id = response.json()["id"]
        assert client.get(f"/api/v1/cases/{case_id}").status_code == 200
        assert client.get("/api/v1/cases").json()[0]["id"] == case_id
        review = {"expected_revision": 1, "issue_id": 1, "action": "accept"}
        assert client.post(f"/api/v1/cases/{case_id}/reviews", json=review).status_code == 200
        assert client.post(f"/api/v1/cases/{case_id}/reviews", json=review).status_code == 409
        download = client.get(f"/api/v1/cases/{case_id}/handoff/1")
        assert download.json()["handoff_created"] is False
        assert (
            download.headers["content-disposition"]
            == f'attachment; filename="handoff-{case_id}-1.json"'
        )
        assert client.get(f"/api/v1/cases/{uuid4()}").status_code == 404
        assert (
            client.post(
                "/api/v1/cases", json={"query": "Complaint", "response": {"sources": []}}
            ).status_code
            == 422
        )
    finally:
        app.dependency_overrides.clear()


def test_working_ethernet_excludes_line_fault_procedures():
    text = "Broadband drops. Ethernet works. All wireless devices disconnect."
    analysis = generated(ConversationRequest(query=text)).issues[0].resolution.analysis
    analysis.products = extract_products(text)
    analysis.reported_facts = extract_facts(text)
    assert supported_scopes(analysis) == {"home_wifi"}


def test_billing_questions_remain_specific_even_with_accepted_category():
    text = "My bill has a duplicate charge."
    questions = clarification_questions(
        accepted=True,
        products=extract_products(text),
        severity=assess_severity(text),
        facts=[],
        requests=[],
    )
    assert questions == ["Which charge or payment is affected, and is it pending or settled?"]


def test_frontend_assets_and_root_are_served_without_inference():
    client = TestClient(app)
    assert client.get("/").status_code == 200
    assert "Turn a complaint" in client.get("/").text
    assert "A clear next step" in client.get("/workspace").text
    assert client.get("/assets/app.js").status_code == 200
    assert client.get("/assets/style.css").status_code == 200
