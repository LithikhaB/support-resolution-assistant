"""Train category baselines on complaints and select using development families only."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from app.config.settings import Settings
from app.ingestion.artifacts import file_sha256, write_json
from app.retrieval.embeddings import get_embedding_service
from app.understanding.augmentation import augment_training
from app.understanding.calibration import select_thresholds
from app.understanding.classifier import CategoryClassifier, ClassifierArtifact, make_vectorizer
from app.understanding.routing import load_category_products


def load_split(directory: Path, split: str) -> list[dict]:
    """Validate a named query artifact and its source checksum without reading other splits."""
    if split not in {"train", "dev", "test"}:
        raise ValueError("Unsupported data split")
    path = directory / f"{split}.jsonl"
    manifest = json.loads((directory / "quality_report.json").read_text(encoding="utf-8"))
    if file_sha256(path) != manifest["file_sha256"][path.name]:
        raise ValueError(f"{split} checksum mismatch; regenerate or review the dataset")
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows or any(
        row.get("split") != split
        or not isinstance(row.get("query"), str)
        or not row["query"].strip()
        or not row.get("scenario_family")
        or not row.get("labels", {}).get("intent")
        for row in rows
    ):
        raise ValueError(f"Invalid {split} examples")
    if len({row["query_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate query IDs")
    return rows


def validate_separation(train: list[dict], dev: list[dict]) -> None:
    """Reject shared scenario families, exact complaints and unseen development labels."""
    if {r["scenario_family"] for r in train} & {r["scenario_family"] for r in dev}:
        raise ValueError("Scenario families overlap across training and development")
    if {r["query"].casefold().strip() for r in train} & {
        r["query"].casefold().strip() for r in dev
    }:
        raise ValueError("Complaints overlap across training and development")
    if not {r["labels"]["intent"] for r in dev} <= {r["labels"]["intent"] for r in train}:
        raise ValueError("Development contains categories missing from training")


def classification_metrics(rows: list[dict], predictions: list[str], classes: list[str]) -> dict:
    """Report variant metrics and family-average accuracy without treating variants as independent."""
    labels = [r["labels"]["intent"] for r in rows]
    families = sorted({r["scenario_family"] for r in rows})
    by_family = {
        family: float(
            np.mean(
                [
                    truth == prediction
                    for row, truth, prediction in zip(rows, labels, predictions, strict=True)
                    if row["scenario_family"] == family
                ]
            )
        )
        for family in families
    }
    return {
        "examples": len(rows),
        "families": len(families),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, labels=classes, average="macro", zero_division=0)
        ),
        "family_mean_accuracy": float(np.mean(list(by_family.values()))),
        "family_accuracy": by_family,
        "class_order": classes,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=classes).tolist(),
    }


def train_classifier(settings: Settings, *, embedder=None, report_path: Path | None = None) -> dict:
    """Compare fixed local feature baselines and publish one validated JSON classifier."""
    train = load_split(settings.corpus_dir, "train")
    base_examples = len(train)
    augmentation_path = settings.corpus_dir / "training_paraphrases.json"
    train, augmentation_hash = augment_training(train, augmentation_path)
    dev = load_split(settings.corpus_dir, "dev")
    validate_separation(train, dev)
    hashes = {
        split: file_sha256(settings.corpus_dir / f"{split}.jsonl") for split in ("train", "dev")
    }
    texts = [r["query"] for r in train]
    dev_texts = [r["query"] for r in dev]
    labels = [r["labels"]["intent"] for r in train]
    family_counts = Counter(r["scenario_family"] for r in train)
    weights = np.asarray([1 / family_counts[r["scenario_family"]] for r in train])
    weights /= weights.mean()
    vectorizer = make_vectorizer()
    lexical_train = vectorizer.fit_transform(texts)
    lexical_dev = vectorizer.transform(dev_texts)
    encoder = embedder or get_embedding_service()
    semantic_train = np.asarray(encoder.embed_documents(texts))
    semantic_dev = np.asarray(encoder.embed_documents(dev_texts))
    candidates = []
    category_products = load_category_products(settings.category_products_path)
    for feature_type, train_features, dev_features in (
        ("tfidf", lexical_train, lexical_dev),
        ("minilm", semantic_train, semantic_dev),
    ):
        for regularization_c in (1.0, 4.0, 16.0):
            model = LogisticRegression(C=regularization_c, max_iter=2000, random_state=42)
            model.fit(train_features, labels, sample_weight=weights)
            if np.any(model.n_iter_ >= model.max_iter):
                raise ValueError("Classifier did not converge; no model was published")
            classes = model.classes_.tolist()
            metrics = classification_metrics(dev, model.predict(dev_features).tolist(), classes)
            metrics["selective_routing"] = select_thresholds(
                dev, model.predict_proba(dev_features), classes, category_products=category_products
            )
            artifact = ClassifierArtifact(
                feature_type=feature_type,
                classes=classes,
                coefficients=model.coef_.tolist(),
                intercept=model.intercept_.tolist(),
                vocabulary=vectorizer.vocabulary_ if feature_type == "tfidf" else {},
                idf=vectorizer.idf_.tolist() if feature_type == "tfidf" else [],
                embedding_model=settings.embedding_model if feature_type == "minilm" else None,
                embedding_revision=settings.tokenizer_revision
                if feature_type == "minilm"
                else None,
                train_sha256=hashes["train"],
                dev_sha256=hashes["dev"],
                augmentation_sha256=augmentation_hash,
                training_families=sorted(family_counts),
                development_families=sorted({r["scenario_family"] for r in dev}),
                sklearn_version=sklearn.__version__,
                regularization_c=regularization_c,
            )
            restored = CategoryClassifier(artifact, settings=settings, embedder=encoder)
            if feature_type == "tfidf":
                restored_scores = restored.probabilities(dev_texts)
            else:
                logits = semantic_dev @ restored.coefficients.T + restored.intercept
                logits -= logits.max(axis=1, keepdims=True)
                restored_scores = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
            if not np.allclose(restored_scores, model.predict_proba(dev_features), atol=1e-7):
                raise ValueError("Exported classifier differs from fitted model")
            candidates.append((artifact, metrics))
    selected, metrics = max(
        candidates,
        key=lambda pair: (
            pair[1]["selective_routing"] is not None,
            pair[1]["macro_f1"],
            pair[0].feature_type == "tfidf",
            -pair[0].regularization_c,
        ),
    )
    if any(
        file_sha256(settings.corpus_dir / f"{split}.jsonl") != value
        for split, value in hashes.items()
    ):
        raise ValueError("Dataset changed during training; no model was published")
    if (
        file_sha256(augmentation_path) if augmentation_path.exists() else None
    ) != augmentation_hash:
        raise ValueError("Training paraphrases changed; no model was published")
    write_json(settings.understanding_model_path, selected.model_dump(mode="json"))
    report = {
        "purpose": "Development selection on synthetic scenario families, not real-world accuracy.",
        "selection": "Prefer candidates meeting the unchanged selective-routing accuracy and coverage floors, then highest development macro-F1; ties prefer TF-IDF, then lower C. Without an eligible candidate, routing remains conservative.",
        "test_split_used": False,
        "train_examples": len(train),
        "base_train_examples": base_examples,
        "augmentation_sha256": augmentation_hash,
        "augmentation_note": "AI-authored paraphrases of training families only; no new independent scenarios or test examples.",
        "train_families": len(family_counts),
        "train_sha256": hashes["train"],
        "dev_sha256": hashes["dev"],
        "selected": {
            "feature_type": selected.feature_type,
            "C": selected.regularization_c,
            **metrics,
        },
        "candidates": [
            {"feature_type": artifact.feature_type, "C": artifact.regularization_c, **result}
            for artifact, result in candidates
        ],
        "artifact_sha256": file_sha256(settings.understanding_model_path),
        "routing": {
            "min_score": settings.understanding_min_score,
            "min_margin": settings.understanding_min_margin,
            "note": "Conservative fixed heuristics, not calibrated confidence or proven OOD detection.",
        },
    }
    write_json(report_path or settings.data_dir / "evaluation/understanding_development.json", report)
    return report
