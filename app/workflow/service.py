"""Coordinate saved conversations, agent reviews, reported outcomes and local handoffs."""

from datetime import UTC, datetime
from functools import lru_cache
from uuid import uuid4

from app.config.settings import get_settings
from app.resolution.conversation import ConversationRequest, resolve_conversation
from app.resolution.service import get_resolution_service
from app.workflow.models import CaseRecord, ReviewEvent
from app.workflow.store import CaseStore, RevisionConflict


class WorkflowService:
    """Keep generated evidence immutable while appending human decisions to each case."""

    def __init__(self, store, resolver=None):
        self.store = store
        self.resolver = resolver

    def generate(self, request):
        """Create evidence on the server instead of accepting client-supplied citations."""
        return resolve_conversation(request, self.resolver or get_resolution_service())

    def create(self, request):
        """Persist a case only after the full conversation resolves successfully."""
        response = self.generate(request)
        now = datetime.now(UTC)
        return self.store.create(
            CaseRecord(
                id=uuid4(),
                revision=1,
                generation=1,
                created_at=now,
                updated_at=now,
                request=request,
                response=response,
            )
        )

    def current(self, case_id, expected_revision):
        """Reject stale clients before running expensive model inference."""
        case = self.store.get(case_id)
        if case.revision != expected_revision:
            raise RevisionConflict()
        return case

    def followup(self, case_id, request):
        """Regenerate from saved customer history and invalidate prior review applicability."""
        case = self.current(case_id, request.expected_revision)
        payload = case.request.model_dump()
        payload["turns"].append(request.turn.model_dump())
        updated = ConversationRequest.model_validate(payload)
        response = self.generate(updated)
        case.request, case.response = updated, response
        case.generation += 1
        case.revision += 1
        case.updated_at = datetime.now(UTC)
        return self.store.save(case, request.expected_revision)

    def review(self, case_id, request):
        """Record review intent; editing never inherits machine citation validation."""
        case = self.current(case_id, request.expected_revision)
        issue = next((i for i in case.response.issues if i.issue_id == request.issue_id), None)
        if issue is None:
            raise ValueError("issue_id does not exist")
        if request.action == "accept" and issue.resolution.validation.status != "passed":
            raise ValueError("a draft with failed validation cannot be accepted")
        if len(case.reviews) >= 100:
            raise ValueError("case review history is full")
        case.revision += 1
        case.updated_at = datetime.now(UTC)
        case.reviews.append(
            ReviewEvent(
                revision=case.revision,
                generation=case.generation,
                created_at=case.updated_at,
                review=request,
                reviewed_text=issue.resolution.draft
                if request.action == "accept"
                else request.edited_draft,
                original_resolution=issue.resolution.model_copy(deep=True),
                text_validation={
                    "accept": "original_contract",
                    "edit": "agent_edit_not_validated",
                    "reject": "rejected",
                }[request.action],
            )
        )
        return self.store.save(case, request.expected_revision)

    def handoff(self, case_id, issue_id):
        """Prepare a structured download without contacting a specialist or promising a repair."""
        case = self.store.get(case_id)
        issue = next((i for i in case.response.issues if i.issue_id == issue_id), None)
        if issue is None:
            raise KeyError(issue_id)
        response = issue.resolution
        latest = next(
            (
                r
                for r in reversed(case.reviews)
                if r.generation == case.generation and r.review.issue_id == issue_id
            ),
            None,
        )
        products = {p.product for p in response.analysis.products}
        target = response.decision.target
        if target == "support_agent":
            target = (
                "billing_team"
                if "billing" in products
                else "mobile_support"
                if "mobile" in products
                else "technical_support"
            )
        return {
            "case_id": str(case.id),
            "revision": case.revision,
            "issue_id": issue_id,
            "handoff_created": False,
            "delivery": "local_draft_only",
            "target": target,
            "complaint": issue.complaint,
            "priority": response.decision.priority,
            "reported_facts": response.analysis.reported_facts,
            "attempted_actions": response.acknowledged_actions,
            "open_questions": response.clarification_questions,
            "sources": response.sources,
            "historical_cases": response.historical_cases,
            "language_draft": response.language_draft,
            "language_status": response.language_status,
            "conditional_suggestions": response.suggestions,
            "agent_review": latest,
            "limitations": response.limitations,
        }


@lru_cache(maxsize=1)
def get_workflow_service():
    """Reuse a lightweight service; each store operation opens its own connection."""
    return WorkflowService(CaseStore(get_settings().case_store_path))
