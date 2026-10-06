"""Authenticated live corpus updates; retrieval and category prediction are distinct."""

import secrets

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.config.settings import get_settings
from app.database.connection import get_connection
from app.ingestion.live import PREDICTION_NOTE, IngestValidationError, ingest_documents
from app.ingestion.schema import SupportDocument
from app.monitoring.metrics import ingest_documents_added
from app.retrieval.cache import published_revision
from app.understanding.service import get_understanding_service

router = APIRouter(prefix="/api/v1", tags=["ingestion"])


class IngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    documents: list[SupportDocument] = Field(min_length=1, max_length=20)


def require_admin(x_admin_key: str = Header(default="", alias="X-Admin-Key")):
    configured = get_settings().ingest_admin_key.get_secret_value()
    if not configured or not secrets.compare_digest(configured.encode(), x_admin_key.encode()):
        raise HTTPException(403, "Ingestion is disabled or unauthorized.")


@router.post("/ingest", dependencies=[Depends(require_admin)])
def ingest(request: IngestRequest):
    """Update retrieval immediately. New category prediction needs labeled data/mapping,
    python -m scripts train && python -m scripts calibrate; ingestion does not retrain.
    """
    try:
        result = ingest_documents(request.documents)
    except IngestValidationError:
        raise HTTPException(
            422, "Invalid corpus update: check unique IDs, split and KB references."
        ) from None
    except Exception:
        raise HTTPException(
            503,
            "Ingestion unavailable; the update was not published.",
            headers={"Retry-After": "5"},
        ) from None
    ingest_documents_added(result["added"])
    return result


@router.get("/categories")
def categories():
    """List deployed classifier classes and live document counts, including untrained intents."""
    try:
        classes = sorted(get_understanding_service().classifier.artifact.classes)
        published_revision()
        with get_connection() as conn:
            counts = dict(
                conn.execute(
                    "SELECT intent,count(*) FROM documents WHERE intent IS NOT NULL GROUP BY intent ORDER BY intent"
                ).fetchall()
            )
        return {
            "classifier_classes": classes,
            "document_counts": counts,
            "category_prediction_note": PREDICTION_NOTE,
        }
    except Exception:
        raise HTTPException(
            503, "Categories unavailable; check classifier and index readiness."
        ) from None
