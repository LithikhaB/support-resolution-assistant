from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

MIN_BODY_CHARS = 15


class DocType(str, Enum):
    HISTORICAL_RESPONSE = "historical_response"
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
    """Unified record for historical replies, verified resolutions and KB articles.

    Free-string intent and product fields allow new labels in storage.
    Recognizing evolving categories still requires definitions and evaluation.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    doc_id: str = Field(min_length=1)
    doc_type: DocType
    title: str = Field(min_length=1)
    body: str = Field(min_length=MIN_BODY_CHARS)
    resolution: str | None = None
    response: str | None = None
    outcome_status: str = Field(
        default="unknown", pattern="^(unknown|verified_resolved|simulated_resolved)$"
    )
    ticket_type: str | None = None
    priority: Severity | None = None
    intent: str | None = None
    product: str | None = None
    severity: Severity | None = None
    sentiment: Sentiment | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_evidence(self) -> "SupportDocument":
        """Enforce distinct real, simulated, unknown and knowledge-base evidence semantics."""
        if self.doc_type == DocType.RESOLVED_TICKET:
            if not self.resolution or self.outcome_status not in {
                "verified_resolved",
                "simulated_resolved",
            }:
                raise ValueError(
                    "Resolved tickets require a resolution and an explicit resolved outcome"
                )
            if self.outcome_status == "simulated_resolved":
                if self.metadata.get("is_synthetic") is not True or not self.metadata.get(
                    "scenario_family"
                ):
                    raise ValueError(
                        "Simulated resolutions require synthetic provenance and scenario_family"
                    )
            elif self.metadata.get("is_synthetic"):
                raise ValueError("Synthetic tickets cannot claim verified real-world resolution")
            evidence = self.metadata.get("outcome_evidence")
            if not isinstance(evidence, str) or not evidence.strip():
                raise ValueError("Resolved tickets require outcome_evidence provenance")
        elif self.doc_type == DocType.HISTORICAL_RESPONSE:
            if not self.response:
                raise ValueError("Historical responses require a non-empty response")
            if self.resolution is not None or self.outcome_status != "unknown":
                raise ValueError("Historical responses cannot claim a verified resolution")
        elif self.outcome_status != "unknown":
            raise ValueError("Knowledge-base articles do not have ticket outcomes")
        return self
