"""Expose conditional evidence-based drafts with sanitized dependency errors."""

import logging

import psycopg
from fastapi import APIRouter, Depends, HTTPException

from app.resolution.conversation import (
    ConversationRequest,
    ConversationResponse,
    resolve_conversation,
)
from app.resolution.models import ResolutionRequest, ResolutionResponse
from app.resolution.service import ResolutionService, get_resolution_service
from app.retrieval.embeddings import EmbeddingInputTooLong
from app.retrieval.vector_search import RetrievalUnavailable
from app.understanding.classifier import UnderstandingUnavailable

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
    request: ConversationRequest, service: ResolutionService = Depends(get_resolution_service)
):
    """Replay customer replies and return isolated drafts for each explicitly stated issue."""
    return run_request(lambda: resolve_conversation(request, service))


def run_request(operation):
    """Translate infrastructure errors consistently across single and multi-turn resolution."""
    try:
        return operation()
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
