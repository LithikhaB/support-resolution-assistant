"""Versioned case and review contracts for a local single-user demonstration."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.resolution.conversation import ConversationRequest, ConversationResponse, CustomerTurn
from app.resolution.models import ResolutionResponse


class RevisionRequest(BaseModel):
    """Prevent stale tabs from overwriting newer work."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    expected_revision: int = Field(ge=1, strict=True)


class FollowupRequest(RevisionRequest):
    """Append one validated customer turn to the persisted history."""

    turn: CustomerTurn


class ReviewRequest(RevisionRequest):
    """Record an agent decision and explicitly reported outcome, without executing actions."""

    issue_id: int = Field(ge=1, le=4, strict=True)
    action: Literal["accept", "edit", "reject"]
    edited_draft: str | None = Field(default=None, min_length=1, max_length=16000)
    notes: str = Field(default="", max_length=2000)
    outcome: Literal["pending", "resolved", "unresolved", "escalated"] = "pending"
    outcome_notes: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def require_review_details(self):
        """Make edits, rejection and claimed outcomes explainable in the audit trail."""
        if (self.action == "edit") != (self.edited_draft is not None):
            raise ValueError("edited_draft is required only for edit decisions")
        if self.action in {"edit", "reject"} and not self.notes:
            raise ValueError("explain edited or rejected drafts in notes")
        if self.outcome != "pending" and not self.outcome_notes:
            raise ValueError("record outcome evidence in outcome_notes")
        return self


class ReviewEvent(BaseModel):
    """Retain immutable decisions tied to the exact generated response reviewed."""

    revision: int
    generation: int
    created_at: datetime
    review: ReviewRequest
    reviewed_text: str | None
    original_resolution: ResolutionResponse
    text_validation: Literal["original_contract", "agent_edit_not_validated", "rejected"]


class CaseRecord(BaseModel):
    """Persist input, generated evidence and all historical reviews separately."""

    id: UUID
    revision: int
    generation: int
    created_at: datetime
    updated_at: datetime
    request: ConversationRequest
    response: ConversationResponse
    reviews: list[ReviewEvent] = Field(default_factory=list)
