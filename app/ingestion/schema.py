from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

MIN_BODY_CHARS = 15


class DocType(str, Enum):
    RESOLVED_TICKET = "resolved_ticket"
    KNOWLEDGE_BASE = "knowledge_base"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Sentiment(str, Enum):
    NEUTRAL = "neutral"
    CONCERNED = "concerned"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"


class SupportDocument(BaseModel):
    """Unified record for resolved tickets and KB articles.

    `intent` and `product` are free strings on purpose: ticket classes evolve,
    and new classes must not require a code change or model retraining.
    """

    doc_id: str = Field(min_length=1)
    doc_type: DocType
    title: str = Field(min_length=1)
    body: str = Field(min_length=MIN_BODY_CHARS)
    resolution: str | None = None
    intent: str | None = None
    product: str | None = None
    severity: Severity | None = None
    sentiment: Sentiment | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)