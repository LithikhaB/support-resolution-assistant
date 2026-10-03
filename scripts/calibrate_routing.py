"""Select category acceptance thresholds on development families without using test queries."""

from collections import defaultdict

import numpy as np

from app.config.settings import get_settings
from app.ingestion.artifacts import digest, file_sha256, write_json
from app.understanding.classifier import CategoryClassifier
from app.understanding.routing import RoutingPolicy, compatible_category
from app.understanding.signals import extract_products
from app.understanding.training import load_split


def select_thresholds(
    rows, scores, classes, *, target_accuracy=0.85, minimum_families=5, minimum_examples=20
):
    """Maximize accepted coverage subject to example and family-average accuracy floors."""
    labels = np.asarray([r["labels"]["intent"] for r in rows])
    order = np.argsort(-scores, axis=1, kind="stable")
    best = scores[np.arange(len(scores)), order[:, 0]]
    margins = best - scores[np.arange(len(scores)), order[:, 1]]
    correct = np.asarray([classes[i] for i in order[:, 0]]) == labels
    compatible = np.asarray(
        [
            compatible_category(classes[index], extract_products(row["query"]))
            for row, index in zip(rows, order[:, 0], strict=True)
        ]
    )
    choices = []
    for threshold in np.arange(0.10, 0.51, 0.01):
        for margin in np.arange(0.00, 0.21, 0.01):
            selected = (best >= threshold) & (margins >= margin) & compatible
            families = defaultdict(list)
            for row, accepted, hit in zip(rows, selected, correct, strict=True):
                if accepted:
                    families[row["scenario_family"]].append(bool(hit))
            if selected.sum() < minimum_examples or len(families) < minimum_families:
                continue
            accuracy = float(correct[selected].mean())
            family_accuracy = float(np.mean([np.mean(values) for values in families.values()]))
            if min(accuracy, family_accuracy) < target_accuracy:
                continue
            choices.append(
                {
                    "min_score": round(float(threshold), 2),
                    "min_margin": round(float(margin), 2),
                    "accepted": int(selected.sum()),
                    "accepted_families": len(families),
                    "accepted_accuracy": accuracy,
                    "family_mean_accuracy": family_accuracy,
                    "total": len(rows),
                }
            )
    return (
        max(
            choices,
            key=lambda c: (
                c["accepted"],
                c["family_mean_accuracy"],
                c["min_margin"],
                c["min_score"],
            ),
        )
        if choices
        else None
    )


def main():
    """Publish a model-bound routing profile only when the prespecified criterion is met."""
    settings = get_settings()
    classifier = CategoryClassifier.load(settings.understanding_model_path, settings=settings)
    rows = load_split(settings.corpus_dir, "dev")
    dev_hash = file_sha256(settings.corpus_dir / "dev.jsonl")
    if dev_hash != classifier.artifact.dev_sha256:
        raise ValueError("Development data differs from the classifier artifact")
    if {r["scenario_family"] for r in rows} & set(classifier.artifact.training_families):
        raise ValueError("Calibration families overlap classifier training")
    scores = classifier.probabilities([r["query"] for r in rows])
    chosen = select_thresholds(rows, scores, classifier.artifact.classes)
    if chosen is None:
        raise SystemExit(
            "No development routing policy met the accuracy and coverage criteria; existing profile unchanged."
        )
    if file_sha256(settings.corpus_dir / "dev.jsonl") != dev_hash:
        raise ValueError("Development data changed during calibration")
    profile = RoutingPolicy(
        model_version=digest(classifier.artifact.model_dump(mode="json"))[:16],
        dev_sha256=dev_hash,
        **chosen,
        selection_note="Development-selected thresholds; >=85% accepted-example and accepted-family mean accuracy, >=5 families and >=20 examples. Compatible mentioned service required. Same development split selected the model, so these figures are optimistic selection results, not independent accuracy or proven OOD detection.",
    )
    write_json(settings.understanding_routing_path, profile.model_dump(mode="json"))
    write_json(
        settings.data_dir / "evaluation/routing_development_v2.json",
        {
            **profile.model_dump(mode="json"),
            "test_split_used": False,
            "classifier_sha256": file_sha256(settings.understanding_model_path),
        },
    )
    print(profile.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
