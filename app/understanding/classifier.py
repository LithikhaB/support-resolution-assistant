"""Load validated JSON coefficients and score local category-classifier features."""

import json
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.feature_extraction.text import TfidfVectorizer

from app.config.settings import Settings, get_settings
from app.retrieval.embeddings import EmbeddingInputTooLong, get_embedding_service
from app.understanding.models import CategoryCandidate


class UnderstandingUnavailable(RuntimeError):
    """The local classifier is missing, invalid or incompatible with its encoder."""


class ClassifierArtifact(BaseModel):
    """Persist numerical weights without executable pickle serialization."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    version: Literal[1] = 1
    feature_type: Literal["tfidf", "minilm"]
    classes: list[str] = Field(min_length=3)
    coefficients: list[list[float]]
    intercept: list[float]
    vocabulary: dict[str, int] = Field(default_factory=dict)
    idf: list[float] = Field(default_factory=list)
    embedding_model: str | None = None
    embedding_revision: str | None = None
    train_sha256: str
    dev_sha256: str
    augmentation_sha256: str | None = None
    training_families: list[str]
    development_families: list[str]
    sklearn_version: str
    regularization_c: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_dimensions(self):
        """Reject malformed feature mappings, split leakage and invalid weight dimensions."""
        if len(set(self.classes)) != len(self.classes) or any(not x.strip() for x in self.classes):
            raise ValueError("Classifier classes must be unique and non-empty")
        if set(self.training_families) & set(self.development_families):
            raise ValueError("Training and development families overlap")
        dimension = len(self.vocabulary) if self.feature_type == "tfidf" else 384
        if dimension == 0 or len(self.coefficients) != len(self.classes):
            raise ValueError("Invalid classifier matrix shape")
        if any(len(row) != dimension for row in self.coefficients) or len(self.intercept) != len(
            self.classes
        ):
            raise ValueError("Invalid classifier feature dimensions")
        if self.feature_type == "tfidf":
            if set(self.vocabulary.values()) != set(range(dimension)) or len(self.idf) != dimension:
                raise ValueError("Invalid vocabulary or IDF mapping")
            if any(value <= 0 for value in self.idf):
                raise ValueError("IDF weights must be positive")
        elif not self.embedding_model or not self.embedding_revision:
            raise ValueError("Embedding model and revision are required")
        return self


def make_vectorizer(vocabulary: dict[str, int] | None = None) -> TfidfVectorizer:
    """Use identical word/bigram TF-IDF settings for fitting and inference."""
    return TfidfVectorizer(
        ngram_range=(1, 2), sublinear_tf=True, max_features=12000, vocabulary=vocabulary
    )


class CategoryClassifier:
    """Predict categories with the saved local encoder and linear decision weights."""

    def __init__(
        self, artifact: ClassifierArtifact, *, settings: Settings | None = None, embedder=None
    ):
        self.artifact = artifact
        self.settings = settings or get_settings()
        self.embedder = embedder
        self.coefficients = np.asarray(artifact.coefficients)
        self.intercept = np.asarray(artifact.intercept)
        self.vectorizer = None
        if artifact.feature_type == "tfidf":
            self.vectorizer = make_vectorizer(artifact.vocabulary)
            self.vectorizer.idf_ = np.asarray(artifact.idf)
        elif (
            artifact.embedding_model != self.settings.embedding_model
            or artifact.embedding_revision != self.settings.tokenizer_revision
        ):
            raise UnderstandingUnavailable("Classifier and configured embedding revisions differ")

    def probabilities(self, texts: list[str]) -> np.ndarray:
        """Return stable multiclass softmax scores using the original feature contract."""
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("Provide at least one non-empty complaint")
        if self.vectorizer is not None:
            features = self.vectorizer.transform(texts)
        else:
            encoder = self.embedder or get_embedding_service()
            try:
                features = np.asarray(encoder.embed_documents(texts))
            except EmbeddingInputTooLong:
                # Keep every segment of long complaints; do not silently truncate.
                features = np.asarray([encoder.embed_query(text) for text in texts])
        logits = np.asarray(features @ self.coefficients.T) + self.intercept
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(logits)
        return weights / weights.sum(axis=1, keepdims=True)

    def predict(self, query: str) -> list[CategoryCandidate]:
        """Return three likely categories with deterministic ordering for tied scores."""
        scores = self.probabilities([query])[0]
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], self.artifact.classes[i]))[:3]
        return [
            CategoryCandidate(category=self.artifact.classes[i], score=float(scores[i]))
            for i in order
        ]

    @classmethod
    def load(cls, path: Path, *, settings: Settings | None = None):
        """Load a JSON artifact and convert expected artifact failures to one service error."""
        try:
            artifact = ClassifierArtifact.model_validate_json(path.read_text(encoding="utf-8"))
            return cls(artifact, settings=settings)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise UnderstandingUnavailable(
                "Train or restore a valid local classifier artifact"
            ) from exc
