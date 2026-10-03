"""Local case and agent review endpoints with sanitized storage errors."""

import sqlite3
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.api.resolution import run_request
from app.resolution.conversation import ConversationRequest
from app.workflow.models import CaseRecord, FollowupRequest, ReviewRequest
from app.workflow.service import get_workflow_service
from app.workflow.store import RevisionConflict

router = APIRouter(prefix="/api/v1/cases", tags=["agent workflow"])


def execute(operation):
    """Keep stale updates, missing records and infrastructure failures distinguishable."""
    try:
        return run_request(operation)
    except RevisionConflict:
        raise HTTPException(409, "Case changed. Reload it before submitting again.") from None
    except KeyError:
        raise HTTPException(404, "Case or issue not found.") from None
    except ValidationError:
        raise HTTPException(422, "Invalid or over-budget conversation update.") from None
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    except sqlite3.Error:
        raise HTTPException(503, "Local case storage is unavailable.") from None


@router.get("")
def recent(service=Depends(get_workflow_service)):
    """List the thirty most recently updated local cases."""
    return execute(service.store.recent)


@router.post("", response_model=CaseRecord, status_code=201)
def create(request: ConversationRequest, service=Depends(get_workflow_service)):
    """Generate and save a new case without dispatching its draft."""
    return execute(lambda: service.create(request))


@router.get("/{case_id}", response_model=CaseRecord)
def get_case(case_id: UUID, service=Depends(get_workflow_service)):
    """Reload the saved response and review history without rerunning models."""
    return execute(lambda: service.store.get(case_id))


@router.post("/{case_id}/followups", response_model=CaseRecord)
def followup(case_id: UUID, request: FollowupRequest, service=Depends(get_workflow_service)):
    """Append one reply to a versioned saved conversation."""
    return execute(lambda: service.followup(case_id, request))


@router.post("/{case_id}/reviews", response_model=CaseRecord)
def review(case_id: UUID, request: ReviewRequest, service=Depends(get_workflow_service)):
    """Record acceptance, an edit or rejection and an optional reported outcome."""
    return execute(lambda: service.review(case_id, request))


@router.get("/{case_id}/handoff/{issue_id}")
def handoff(case_id: UUID, issue_id: int, service=Depends(get_workflow_service)):
    """Export an advisory handoff package; no external system receives it."""
    package = execute(lambda: service.handoff(case_id, issue_id))
    return JSONResponse(
        jsonable_encoder(package),
        headers={
            "Content-Disposition": f'attachment; filename="handoff-{case_id}-{issue_id}.json"'
        },
    )
