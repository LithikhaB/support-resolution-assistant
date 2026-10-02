"""Expose local complaint analysis with bounded inputs and sanitized failures."""

import logging

from fastapi import APIRouter, HTTPException

from app.retrieval.embeddings import EmbeddingInputTooLong
from app.understanding.classifier import UnderstandingUnavailable
from app.understanding.models import AnalysisResponse, AnalyzeRequest
from app.understanding.service import get_understanding_service

router = APIRouter(prefix="/api/v1", tags=["understanding"])
logger = logging.getLogger(__name__)


@router.post("/analyze", response_model=AnalysisResponse)
def analyze(request: AnalyzeRequest) -> AnalysisResponse:
    """Analyze a complaint without requiring PostgreSQL or a hosted LLM."""
    try:
        return get_understanding_service().analyze(request)
    except EmbeddingInputTooLong:
        raise HTTPException(
            422, "Query exceeds the local embedding token budget; shorten the complaint."
        ) from None
    except (UnderstandingUnavailable, OSError) as exc:
        logger.warning("Understanding unavailable error_type=%s", type(exc).__name__)
        raise HTTPException(
            503,
            "Local understanding model unavailable; run scripts.train_understanding and restart the API.",
        ) from None
