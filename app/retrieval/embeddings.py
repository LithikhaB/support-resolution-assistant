"""CPU embeddings with one shared model per process and explicit input limits."""

import logging
from functools import lru_cache
from threading import Lock
from time import perf_counter
from typing import Any

import numpy as np

from app.config.settings import Settings, get_settings
from app.retrieval.tokenizer import load_tokenizer

logger = logging.getLogger(__name__)
_factory_lock = Lock()


class EmbeddingInputTooLong(ValueError):
    """Input exceeds the embedding model's token budget."""


class EmbeddingService:
    """Reuse a pinned CPU model and validate all embedding inputs and outputs."""

    def __init__(self, settings: Settings, *, model: Any = None, tokenizer: Any = None):
        self.settings = settings
        self.batch_size = settings.embedding_batch_size
        if settings.embedding_dim != 384:
            raise ValueError("The current database requires 384-dimensional embeddings")
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(
                settings.embedding_model,
                device="cpu",
                revision=settings.tokenizer_revision,
                cache_folder=str(settings.data_dir / "models"),
                local_files_only=settings.embedding_local_files_only,
                trust_remote_code=False,
            )
        dimension = getattr(model, "get_embedding_dimension", None)
        if dimension is None:
            dimension = model.get_sentence_embedding_dimension
        if dimension() != settings.embedding_dim:
            raise ValueError("Model dimension does not match the configured database dimension")
        if tokenizer is None:
            tokenizer, _ = load_tokenizer(
                settings.embedding_model,
                settings.tokenizer_revision,
                str(settings.data_dir / "models"),
                settings.embedding_local_files_only,
            )
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = min(256, model.max_seq_length)
        self._encode_lock = Lock()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Validate token budgets and produce normalized CPU vectors in bounded batches."""
        if not isinstance(texts, list):
            raise ValueError("texts must be a list of strings")
        if not texts:
            return []
        for index, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Input {index} must be a non-empty string")
            if len(self.tokenizer.encode(text).ids) > self.max_tokens:
                raise EmbeddingInputTooLong(
                    f"Input {index} exceeds {self.max_tokens} tokens; chunk it before embedding"
                )
        started = perf_counter()
        output: list[list[float]] = []

        with self._encode_lock:
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                vectors = np.asarray(
                    self.model.encode(
                        batch,
                        batch_size=self.batch_size,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                    ),
                    dtype=np.float32,
                )
                if vectors.shape != (len(batch), self.settings.embedding_dim):
                    raise ValueError("Embedding output has an unexpected shape")
                if not np.isfinite(vectors).all():
                    raise ValueError("Embedding output contains non-finite values")
                norms = np.linalg.norm(vectors.astype(np.float64), axis=1, keepdims=True)
                if np.any(norms <= 1e-12):
                    raise ValueError("Embedding output contains a zero vector")
                output.extend((vectors / norms).tolist())
        logger.info(
            "Embedding completed inputs=%d batches=%d dimension=%d elapsed_ms=%.1f",
            len(texts),
            (len(texts) + self.batch_size - 1) // self.batch_size,
            self.settings.embedding_dim,
            (perf_counter() - started) * 1000,
        )
        return output

    def embed_query(self, text: str) -> list[float]:
        """Embed one query using the same contract as indexed documents."""
        return self.embed_documents([text])[0]


@lru_cache(maxsize=1)
def _cached_service() -> EmbeddingService:
    return EmbeddingService(get_settings())


def get_embedding_service() -> EmbeddingService:
    """Lazily initialize one embedding model under a process-local lock."""
    with _factory_lock:
        return _cached_service()
