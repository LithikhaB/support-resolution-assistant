"""Retrieval endpoint: ranked historical evidence, without answer generation."""
import logging

import psycopg
from fastapi import APIRouter, Depends, HTTPException

from app.retrieval.embeddings import EmbeddingInputTooLong
from app.retrieval.models import SearchRequest, SearchResponse
from app.retrieval.service import RetrievalService, get_retrieval_service
from app.retrieval.vector_search import RetrievalUnavailable

router = APIRouter(prefix='/api/v1', tags=['retrieval'])
logger = logging.getLogger(__name__)


@router.post('/retrieve', response_model=SearchResponse)
def retrieve(request: SearchRequest, service: RetrievalService = Depends(get_retrieval_service)):
    try:
        return service.search(request)
    except EmbeddingInputTooLong:
        raise HTTPException(422, 'Query exceeds the embedding token limit; shorten the complaint.') from None
    except psycopg.errors.QueryCanceled:
        logger.warning('Retrieval failed reason=database_timeout')
        raise HTTPException(504, 'Retrieval timed out; retry or narrow the filters.') from None
    except (RetrievalUnavailable, psycopg.Error, OSError) as exc:
        # Exception messages can contain database credentials or local paths.
        logger.warning('Retrieval unavailable error_type=%s', type(exc).__name__)
        raise HTTPException(503, 'Retrieval unavailable; check the database, index and local model cache.',
                            headers={'Retry-After': '5'}) from None
