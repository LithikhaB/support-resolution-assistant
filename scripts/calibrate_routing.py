"""Select category acceptance thresholds on development families without using test queries."""

import argparse
from pathlib import Path

from app.config.settings import get_settings
from app.ingestion.artifacts import digest, file_sha256, write_json
from app.understanding.calibration import select_thresholds
from app.understanding.classifier import CategoryClassifier
from app.understanding.routing import RoutingPolicy, load_category_products
from app.understanding.training import load_split


def main():
    """Publish a model-bound routing profile only when the prespecified criterion is met."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Separate report path for isolated experiments")
    args = parser.parse_args()
    classifier = CategoryClassifier.load(settings.understanding_model_path, settings=settings)
    rows = load_split(settings.corpus_dir, "dev")
    dev_hash = file_sha256(settings.corpus_dir / "dev.jsonl")
    if dev_hash != classifier.artifact.dev_sha256:
        raise ValueError("Development data differs from the classifier artifact")
    if {r["scenario_family"] for r in rows} & set(classifier.artifact.training_families):
        raise ValueError("Calibration families overlap classifier training")
    scores = classifier.probabilities([r["query"] for r in rows])
    chosen = select_thresholds(
        rows,
        scores,
        classifier.artifact.classes,
        category_products=load_category_products(settings.category_products_path),
    )
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
        args.report or settings.data_dir / "evaluation/routing_development_v2.json",
        {
            **profile.model_dump(mode="json"),
            "test_split_used": False,
            "classifier_sha256": file_sha256(settings.understanding_model_path),
        },
    )
    print(profile.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
