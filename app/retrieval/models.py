"""Validated retrieval inputs and evidence-bearing vector results."""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.ingestion.schema import DocType, Severity


class RetrievalFilters(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    doc_type: DocType | None = None
    intent: str | None = Field(default=None, min_length=1, max_length=100)
    severity: Severity | None = None
    ticket_type: str | None = Field(default=None, min_length=1, max_length=100)
    queue: str | None = Field(default=None, min_length=1, max_length=100)


class RetrievalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    query: str = Field(min_length=1, max_length=10000)
    top_k: int = Field(default=10, ge=1, le=100, strict=True)
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)


class EvidenceResult(BaseModel):
    chunk_id: int
    doc_id: str
    chunk_index: int
    content: str
    title: str
    doc_type: DocType
    response: str | None
    resolution: str | None
    outcome_status: str
    metadata: dict[str, Any]


class VectorResult(EvidenceResult):
    cosine_distance: float = Field(allow_inf_nan=False)
    cosine_similarity: float = Field(allow_inf_nan=False)
    vector_rank: int
    sources: list[Literal["vector"]] = Field(default_factory=lambda: ["vector"])


class BM25Result(EvidenceResult):
    bm25_score: float = Field(allow_inf_nan=False)
    bm25_rank: int
    sources: list[Literal["bm25"]] = Field(default_factory=lambda: ["bm25"])


class HybridResult(EvidenceResult):
    rrf_score: float
    vector_contribution: float = 0.0
    bm25_contribution: float = 0.0
    hybrid_rank: int
    vector_rank: int | None = None
    bm25_rank: int | None = None
    cosine_similarity: float | None = None
    bm25_score: float | None = None
    sources: list[Literal["vector", "bm25"]]


class SearchRequest(RetrievalRequest):
    top_k: int = Field(default=5, ge=1, le=100, strict=True)
    mode: Literal['bm25', 'vector', 'hybrid'] = 'hybrid'
    candidate_k: int | None = Field(default=None, ge=1, le=100, strict=True)

    @model_validator(mode='after')
    def validate_candidates(self):
        if self.candidate_k is not None:
            if self.mode != 'hybrid':
                raise ValueError('candidate_k applies only to hybrid search')
            if self.candidate_k < self.top_k:
                raise ValueError('candidate_k must be at least top_k')
        return self


class SearchResponse(BaseModel):
    mode: Literal['bm25', 'vector', 'hybrid']
    results: list[HybridResult | VectorResult | BM25Result]
    elapsed_ms: float = Field(ge=0, allow_inf_nan=False)
