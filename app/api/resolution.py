"""Expose conditional evidence-based drafts with sanitized dependency errors."""

import logging
import re
import secrets
from hashlib import sha256
from typing import Literal
from uuid import UUID

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from app.config.settings import get_settings
from app.resolution.conversation import (
    ConversationRequest,
    ConversationResponse,
    resolve_conversation,
)
from app.resolution.memory import ConversationConflict, ConversationStore, history_context
from app.resolution.models import ResolutionRequest, ResolutionResponse
from app.resolution.service import ResolutionService, get_resolution_service
from app.retrieval.cache import published_revision
from app.retrieval.embeddings import EmbeddingInputTooLong, get_embedding_service
from app.retrieval.vector_search import RetrievalUnavailable
from app.understanding.classifier import UnderstandingUnavailable
from app.understanding.scope import scope_assessment, split_issues

router = APIRouter(prefix="/api/v1", tags=["resolution"])
logger = logging.getLogger(__name__)


@router.post("/resolve", response_model=ResolutionResponse)
def resolve(
    request: ResolutionRequest, service: ResolutionService = Depends(get_resolution_service)
):
    """Return a reviewable agent draft rather than an automatically executed resolution."""
    return run_request(lambda: service.resolve(request))


@router.post("/conversation", response_model=ConversationResponse)
def conversation(
    request: ConversationRequest,
    http_request: Request,
    response: Response,
    service: ResolutionService = Depends(get_resolution_service),
):
    """Replay customer replies and return isolated drafts for each explicitly stated issue."""
    if not get_settings().conversation_storage_enabled or all(
        scope_assessment(part)[0] == "unsupported" for part in split_issues(request.query)
    ):
        return run_request(lambda: resolve_conversation(request, service))
    owner = browser_owner(http_request, response)
    store = ConversationStore()
    if request.conversation_id and not run_request(
        lambda: store.load(owner, request.conversation_id)
    ):
        raise HTTPException(404, "Conversation unavailable in this browser.")

    def operation():
        revision = published_revision()
        token = history_context.set((owner, store, revision))
        try:
            result = resolve_conversation(request, service)
            cid, version = store.save(owner, request, result, revision)
            result.storage = "postgres_redacted"
            result.conversation_id, result.revision = UUID(cid), version
            return result
        finally:
            history_context.reset(token)

    return run_request(operation)


def browser_owner(request, response):
    """Anonymous browser capability; not a substitute for production agent login."""
    if not get_settings().conversation_storage_enabled:
        raise HTTPException(503, "Conversation storage is disabled.")
    origin = request.headers.get("origin")
    if origin and origin != f"{request.url.scheme}://{request.url.netloc}":
        raise HTTPException(403, "Use this application's own page.")
    cookie = request.cookies.get("support_owner", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]{43}", cookie):
        cookie = secrets.token_urlsafe(32)
        response.set_cookie(
            "support_owner",
            cookie,
            httponly=True,
            secure=request.url.scheme == "https",
            samesite="strict",
            max_age=get_settings().conversation_retention_days * 86400,
        )
    return sha256(cookie.encode()).hexdigest()


@router.get("/conversations")
def recent_conversations(request: Request, response: Response):
    owner = browser_owner(request, response)
    return run_request(lambda: {"conversations": ConversationStore().recent(owner)})


@router.get("/conversations/{conversation_id}")
def load_conversation(conversation_id: UUID, request: Request, response: Response):
    row = run_request(
        lambda: ConversationStore().load(browser_owner(request, response), conversation_id)
    )
    if row is None:
        raise HTTPException(404, "Conversation unavailable in this browser.")
    return row


@router.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: UUID, request: Request, response: Response):
    deleted = run_request(
        lambda: ConversationStore().delete(browser_owner(request, response), conversation_id)
    )
    if not deleted:
        raise HTTPException(404, "Conversation unavailable in this browser.")
    return {"deleted": True}


class ReviewedOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    issue_id: int = Field(ge=1, le=4)
    revision: int = Field(ge=1)
    outcome_note: str = Field(min_length=20, max_length=1000)
    confirmation: Literal["simulated_resolution_reviewed"]


@router.post("/conversations/{conversation_id}/review")
def review_outcome(
    conversation_id: UUID, outcome: ReviewedOutcome, request: Request, response: Response
):
    owner = browser_owner(request, response)
    count = run_request(
        lambda: ConversationStore().approve(
            owner,
            conversation_id,
            outcome.issue_id,
            outcome.revision,
            outcome.outcome_note,
            published_revision(),
            get_embedding_service(),
        )
    )
    return {
        "reviewed_procedures": count,
        "outcome_status": "simulated_resolved",
        "scope": "this_browser_only",
    }


def run_request(operation):
    """Translate infrastructure errors consistently across single and multi-turn resolution."""
    try:
        return operation()
    except ConversationConflict as exc:
        raise HTTPException(409, str(exc)) from None
    except EmbeddingInputTooLong:
        raise HTTPException(422, "Complaint exceeds the local token budget; shorten it.") from None
    except psycopg.errors.QueryCanceled:
        raise HTTPException(
            504, "Evidence retrieval timed out; retry or narrow the filters."
        ) from None
    except (UnderstandingUnavailable, RetrievalUnavailable, psycopg.Error, OSError) as exc:
        logger.warning("Resolution unavailable error_type=%s", type(exc).__name__)
        raise HTTPException(
            503,
            "Resolution service unavailable; check the trained classifier, model cache and database index.",
        ) from None
