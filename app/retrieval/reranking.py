"""Bounded local pairwise reranking that preserves original evidence and source ranks."""

from functools import lru_cache
from queue import Queue
from threading import Lock

import numpy as np

from app.config.settings import get_settings
from app.retrieval.embeddings import EmbeddingInputTooLong
from app.retrieval.vector_search import RetrievalUnavailable

_factory_lock = Lock()


class RerankerUnavailable(RetrievalUnavailable):
    """The requested local reranker cannot produce valid relevance scores."""


class RerankingService:
    """Score query and chunk pairs on CPU using a pinned pretrained cross-encoder."""

    def __init__(self, *, settings=None, model=None):
        self.settings = settings or get_settings()
        self._lock = Lock()
        if model is None:
            try:
                from sentence_transformers import CrossEncoder
                from torch.nn import Identity

                model = CrossEncoder(
                    self.settings.reranker_model,
                    revision=self.settings.reranker_revision,
                    device="cpu",
                    max_length=512,
                    activation_fn=Identity(),
                    cache_folder=str(self.settings.data_dir / "models"),
                    local_files_only=self.settings.reranker_local_files_only,
                    trust_remote_code=False,
                )
            except (OSError, ValueError, RuntimeError) as exc:
                raise RerankerUnavailable("Unable to load the local reranker") from exc
        self.model = model

    def rerank(self, query, candidates, top_k):
        """Return copies sorted by finite logits; tied scores retain retrieval order."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be nonempty")
        if type(top_k) is not int or not 1 <= top_k <= 50 or len(candidates) > 50:
            raise ValueError("reranking supports at most 50 candidates and results")
        if len({item.chunk_id for item in candidates}) != len(candidates):
            raise ValueError("reranking candidates must have unique chunk IDs")
        if not candidates:
            return []
        with self._lock:
            try:
                if len(self.model.tokenizer.encode(query, add_special_tokens=True)) > 256:
                    raise EmbeddingInputTooLong("Reranking query exceeds 256 tokens")
                pairs = [(query, item.content) for item in candidates]
                truncated = [
                    len(self.model.tokenizer.encode(a, b, truncation=False)) > 512 for a, b in pairs
                ]
                scores = np.asarray(
                    self.model.predict(
                        pairs,
                        batch_size=self.settings.reranker_batch_size,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                    ),
                    dtype=float,
                )
            except EmbeddingInputTooLong:
                raise
            except (OSError, ValueError, RuntimeError) as exc:
                raise RerankerUnavailable("Unable to score reranking candidates") from exc
        if scores.shape != (len(candidates),) or not np.isfinite(scores).all():
            raise RerankerUnavailable("Invalid reranking scores")
        order = sorted(range(len(candidates)), key=lambda index: (-scores[index], index))
        return [
            candidates[index].model_copy(
                update={
                    "rerank_score": float(scores[index]),
                    "rerank_rank": rank,
                    "rerank_input_truncated": truncated[index],
                }
            )
            for rank, index in enumerate(order[:top_k], 1)
        ]


class RerankerPool:
    """Two independent model/tokenizer instances avoid serializing independent requests."""

    def __init__(self, models):
        self.models = models
        self.settings = models[0].settings
        self.model = models[0].model
        self.available = Queue(maxsize=len(models))
        for model in models:
            self.available.put(model)

    def rerank(self, query, candidates, top_k):
        model = self.available.get()
        try:
            return model.rerank(query, candidates, top_k)
        finally:
            self.available.put(model)


@lru_cache(maxsize=1)
def _cached_reranker():
    settings = get_settings()
    models = [RerankingService(settings=settings) for _ in range(settings.reranker_instances)]
    return models[0] if len(models) == 1 else RerankerPool(models)


def get_reranking_service():
    """Load the reranker only when explicitly requested and reuse it per process."""
    with _factory_lock:
        return _cached_reranker()
