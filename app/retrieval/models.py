"""Validated retrieval inputs and evidence-bearing vector results."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.ingestion.schema import DocType, Severity


class RetrievalFilters(BaseModel):
    """Allow only explicit supported document filters."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    doc_type: DocType | None = None
    intent: str | None = Field(default=None, min_length=1, max_length=100)
    severity: Severity | None = None
    ticket_type: str | None = Field(default=None, min_length=1, max_length=100)
    product: str | None = Field(default=None, min_length=1, max_length=100)
    queue: str | None = Field(default=None, min_length=1, max_length=100)


class RetrievalRequest(BaseModel):
    """Bound query length, result count and supported filters."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    query: str = Field(min_length=1, max_length=10000)
    top_k: int = Field(default=10, ge=1, le=100, strict=True)
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)


class EvidenceResult(BaseModel):
    """Preserve source evidence independently of retrieval ranking scores."""

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
    rerank_score: float | None = Field(default=None, allow_inf_nan=False)
    rerank_rank: int | None = Field(default=None, ge=1)
    rerank_input_truncated: bool = False


class VectorResult(EvidenceResult):
    """Expose cosine distance, similarity and semantic rank alongside evidence."""

    cosine_distance: float = Field(allow_inf_nan=False)
    cosine_similarity: float = Field(allow_inf_nan=False)
    vector_rank: int
    sources: list[Literal["vector"]] = Field(default_factory=lambda: ["vector"])


class BM25Result(EvidenceResult):
    """Expose lexical score and rank alongside evidence."""

    bm25_score: float = Field(allow_inf_nan=False)
    bm25_rank: int
    sources: list[Literal["bm25"]] = Field(default_factory=lambda: ["bm25"])


class HybridResult(EvidenceResult):
    """Expose fused rank and individual source contributions alongside evidence."""

    rrf_score: float = Field(ge=0, allow_inf_nan=False)
    vector_contribution: float = 0.0
    bm25_contribution: float = 0.0
    hybrid_rank: int
    vector_rank: int | None = None
    bm25_rank: int | None = None
    cosine_similarity: float | None = None
    bm25_score: float | None = None
    sources: list[Literal["vector", "bm25"]]


class SearchRequest(RetrievalRequest):
    """Validate mode-specific retrieval options for API and CLI callers."""

    top_k: int = Field(default=5, ge=1, le=100, strict=True)
    mode: Literal["bm25", "vector", "hybrid"] = "hybrid"
    candidate_k: int | None = Field(default=None, ge=1, le=100, strict=True)

    diversify: bool = Field(default=False, strict=True)
    rerank: bool = Field(default=False, strict=True)
    rerank_k: int | None = Field(default=None, ge=1, le=50, strict=True)

    @property
    def ranking_depth(self) -> int:
        """Retrieve a bounded pool before applying optional pairwise reranking."""
        return (self.rerank_k or max(20, self.top_k)) if self.rerank else self.top_k

    @property
    def retrieval_depth(self) -> int:
        """Overfetch for diversity before limiting the pool scored by the reranker."""
        return max(50, self.ranking_depth) if self.diversify else self.ranking_depth

    @model_validator(mode="after")
    def validate_candidates(self):
        """Reject candidate pools that cannot satisfy the requested hybrid result count."""
        if self.rerank_k is not None and not self.rerank:
            raise ValueError("rerank_k requires rerank=true")
        if self.rerank and (self.top_k > 50 or self.ranking_depth < self.top_k):
            raise ValueError("reranking requires top_k <= rerank_k <= 50")
        if self.candidate_k is not None:
            if self.mode != "hybrid":
                raise ValueError("candidate_k applies only to hybrid search")
            if self.candidate_k < self.retrieval_depth:
                raise ValueError("candidate_k must be at least the retrieval depth")
        return self


class SearchResponse(BaseModel):
    """Return ranked evidence and measured request duration."""

    mode: Literal["bm25", "vector", "hybrid"]
    results: list[HybridResult | VectorResult | BM25Result]
    diversified: bool = False
    retrieved_candidates: int = 0
    reranked: bool = False
    reranker_model: str | None = None
    reranker_revision: str | None = None
    reranked_candidates: int = 0
    rerank_score_note: str = (
        "Pairwise relevance scores are not probabilities or proof of a diagnosis."
    )
    elapsed_ms: float = Field(ge=0, allow_inf_nan=False)
