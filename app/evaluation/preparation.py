"""Freeze evaluation inputs and reconstruct a development-selected lexical baseline."""

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from app.ingestion.artifacts import digest, file_sha256, write_json
from app.understanding.augmentation import augment_training
from app.understanding.classifier import CategoryClassifier, make_vectorizer
from app.understanding.training import load_split, validate_separation


def prepare_classifiers(settings):
    """Fit only the preselected TF-IDF baseline on training rows; never inspect test labels."""
    report_path = settings.data_dir / "evaluation/understanding_development.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["artifact_sha256"] != file_sha256(settings.understanding_model_path):
        raise ValueError("Classifier differs from the development-selection report")
    selected = CategoryClassifier.load(settings.understanding_model_path, settings=settings)
    train = load_split(settings.corpus_dir, "train")
    train, augmentation_hash = augment_training(
        train, settings.corpus_dir / "training_paraphrases.json"
    )
    if (
        augmentation_hash != selected.artifact.augmentation_sha256
        or augmentation_hash != report.get("augmentation_sha256")
    ):
        raise ValueError("Classifier training paraphrases changed")
    dev = load_split(settings.corpus_dir, "dev")
    validate_separation(train, dev)
    for split in ("train", "dev"):
        actual = file_sha256(settings.corpus_dir / f"{split}.jsonl")
        if actual != report[f"{split}_sha256"] or actual != getattr(
            selected.artifact, f"{split}_sha256"
        ):
            raise ValueError("Classifier training/development data changed")
    if set(selected.artifact.training_families) != {r["scenario_family"] for r in train} or set(
        selected.artifact.development_families
    ) != {r["scenario_family"] for r in dev}:
        raise ValueError("Classifier family provenance mismatch")
    candidate = max(
        (c for c in report["candidates"] if c["feature_type"] == "tfidf"),
        key=lambda c: (c["macro_f1"], -c["C"]),
    )
    vectorizer = make_vectorizer()
    features = vectorizer.fit_transform([r["query"] for r in train])
    counts = Counter(r["scenario_family"] for r in train)
    weights = np.asarray([1 / counts[r["scenario_family"]] for r in train])
    model = LogisticRegression(C=candidate["C"], max_iter=2000, random_state=42)
    model.fit(
        features, [r["labels"]["intent"] for r in train], sample_weight=weights / weights.mean()
    )
    if np.any(model.n_iter_ >= model.max_iter):
        raise ValueError("Lexical comparison baseline did not converge")
    return selected, vectorizer, model, train, dev


def validate_holdout(rows, train, dev, split):
    """Reject family and exact-text leakage into the held-out evaluation."""
    validate_separation(train, rows)
    if split == "test":
        validate_separation(dev, rows)


def fingerprint(settings, split):
    """Hash code, model, corpus and fixed settings before opening evaluation queries."""
    root = Path(__file__).resolve().parents[2]
    files = sorted(list((root / "app").rglob("*.py")) + list((root / "scripts").glob("*.py")))
    manifest = json.loads((settings.corpus_dir / "quality_report.json").read_text(encoding="utf-8"))
    return {
        "split": split,
        "code_sha256": digest({str(p.relative_to(root)): file_sha256(p) for p in files}),
        "routing_profile_sha256": file_sha256(settings.understanding_routing_path)
        if settings.understanding_routing_path.exists()
        else None,
        "classifier_sha256": file_sha256(settings.understanding_model_path),
        "development_report_sha256": file_sha256(
            settings.data_dir / "evaluation/understanding_development.json"
        ),
        "corpus_sha256": file_sha256(settings.processed_dir / "documents.jsonl"),
        "chunks_sha256": file_sha256(settings.processed_dir / "chunks.jsonl"),
        "declared_query_sha256": manifest["file_sha256"][f"{split}.jsonl"],
        "train_sha256": file_sha256(settings.corpus_dir / "train.jsonl"),
        "augmentation_sha256": file_sha256(settings.corpus_dir / "training_paraphrases.json")
        if (settings.corpus_dir / "training_paraphrases.json").exists()
        else None,
        "dev_sha256": file_sha256(settings.corpus_dir / "dev.jsonl"),
        "challenge_sha256": file_sha256(settings.corpus_dir / "challenge_queries.jsonl"),
        "embedding_model": settings.embedding_model,
        "embedding_revision": settings.tokenizer_revision,
        "reranker_model": settings.reranker_model,
        "reranker_revision": settings.reranker_revision,
        "min_score": settings.understanding_min_score,
        "min_margin": settings.understanding_min_margin,
        "top_k": 5,
        "candidate_k": settings.retrieval_candidate_k,
        "rrf_constant": settings.retrieval_rrf_constant,
        "rerank_k": 20,
        "max_draft_sources": 3,
    }


def freeze_run(output, snapshot):
    """Preserve completed reports and record the policy before evaluating held-out queries."""
    if output.exists():
        raise ValueError(
            "Report already exists; use a new --output path to record an explicit repeat"
        )
    freeze_path = output.with_suffix(".freeze.json")
    if freeze_path.exists():
        existing = json.loads(freeze_path.read_text(encoding="utf-8"))
        if existing["fingerprint"] != snapshot:
            raise ValueError("Inputs changed since the frozen run; use a new --output path")
    else:
        write_json(
            freeze_path,
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "fingerprint": snapshot,
                "note": "Fixed before loading evaluation queries. Do not tune against test results.",
            },
        )
    return freeze_path
