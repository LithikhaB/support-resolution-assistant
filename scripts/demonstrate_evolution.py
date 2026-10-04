"""Demonstrate a new KB article and classifier category in a separate local database."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from app.config.settings import get_settings
from app.database.connection import get_connection
from app.ingestion.artifacts import file_sha256, write_json
from app.ingestion.evolution import prepare_evolution


def active_fingerprint(settings):
    """Protect working corpus, classifier and routing artifacts from experiment writes."""
    paths = [
        settings.processed_dir / "documents.jsonl",
        settings.processed_dir / "chunks.jsonl",
        settings.understanding_model_path,
        settings.understanding_routing_path,
    ]
    fingerprint = {str(path): file_sha256(path) for path in paths}
    with get_connection() as conn:
        fingerprint["active_index"] = list(
            conn.execute(
                "SELECT status,source_hash,chunks_hash,config_hash FROM retrieval_index_state WHERE singleton"
            ).fetchone()
        )
    return fingerprint


def inspect_runtime(query, category, article):
    """Exercise the same local understanding and resolution services as the application."""
    from app.database.connection import get_connection
    from app.database.readiness import check_index
    from app.resolution.models import ResolutionRequest
    from app.resolution.service import get_resolution_service
    from app.understanding.service import get_understanding_service

    classifier = get_understanding_service().classifier
    regression = old_category_regression(classifier, get_settings(), category)
    result = get_resolution_service().resolve(ResolutionRequest(query=query))
    with get_connection(statement_timeout_ms=60000) as conn:
        index = check_index(conn)
    return {
        "new_category": category,
        "class_registered": category in classifier.artifact.classes,
        "top_category": classifier.predict(query)[0].category,
        "new_article_cited": article in {source.doc_id for source in result.sources},
        "citation_status": result.validation.status,
        "index": index,
        "response": result.model_dump(mode="json"),
        "old_category_regression": regression,
    }


def old_category_regression(classifier, settings, new_category):
    """Measure the original development categories separately from the added class."""
    from app.understanding.training import load_split

    rows = [
        row
        for row in load_split(settings.corpus_dir, "dev")
        if row["labels"]["intent"] != new_category
    ]
    values = classifier.probabilities([row["query"] for row in rows])
    predictions = [classifier.artifact.classes[int(value.argmax())] for value in values]
    correct = sum(
        prediction == row["labels"]["intent"]
        for row, prediction in zip(rows, predictions, strict=True)
    )
    categories = {row["labels"]["intent"] for row in rows}
    return {
        "queries": len(rows),
        "categories": len(categories),
        "correct": correct,
        "accuracy": correct / len(rows),
        "all_old_classes_registered": categories <= set(classifier.artifact.classes),
    }


def main():
    """Keep the demo isolated and publish proof only after all preparation steps succeed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture", type=Path, default=Path("data/evolution/dns_category_demo.json")
    )
    parser.add_argument("--output", type=Path, required=True, help="New summary report path")
    parser.add_argument("--inspect", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    if args.output.exists():
        parser.error("choose a new output report")
    if args.inspect:
        write_json(
            args.output,
            inspect_runtime(fixture["query"], fixture["category"], fixture["knowledge"]["doc_id"]),
        )
        return
    settings = get_settings()
    from app.understanding.classifier import CategoryClassifier

    baseline = old_category_regression(
        CategoryClassifier.load(settings.understanding_model_path, settings=settings),
        settings,
        fixture["category"],
    )
    before = active_fingerprint(settings)
    run_id = uuid4().hex[:12]
    root = Path("data/evolution/runs") / run_id
    staged = prepare_evolution(settings, fixture, root)
    database = "support_evolution_" + run_id
    if database == settings.postgres_db:
        raise ValueError("Evolution database must differ from the active database")
    with get_connection() as conn:
        if conn.execute("SELECT 1 FROM pg_database WHERE datname=%s", (database,)).fetchone():
            raise ValueError("Demonstration database already exists; use a fresh run")
    environment = {
        **os.environ,
        "POSTGRES_DB": database,
        "CORPUS_DIR": str((root / "corpus").resolve()),
        "DATA_DIR": str(settings.data_dir.resolve()),
        "UNDERSTANDING_MODEL_PATH": str((root / "classifier.json").resolve()),
        "UNDERSTANDING_ROUTING_PATH": str((root / "routing.json").resolve()),
        "CATEGORY_PRODUCTS_PATH": str((root / "category_products.json").resolve()),
        "LLM_ENABLED": "false",
        "HF_HUB_OFFLINE": "1",
        "EMBEDDING_LOCAL_FILES_ONLY": "true",
        "TOKENIZER_LOCAL_FILES_ONLY": "true",
        "RERANKER_LOCAL_FILES_ONLY": "true",
    }
    inspected = root / "inspection.json"
    commands = [
        ("scripts.setup_database", []),
        ("scripts.chunk_documents", []),
        ("scripts.index_documents", []),
        ("scripts.train_understanding", ["--report", str(root / "development.json")]),
        ("scripts.calibrate_routing", ["--report", str(root / "calibration.json")]),
        (
            "scripts.demonstrate_evolution",
            ["--inspect", "--fixture", str(args.fixture), "--output", str(inspected)],
        ),
    ]
    try:
        for index, (module, arguments) in enumerate(commands):
            print(f"Evolution: {module}", flush=True)
            with (root / f"{index}.log").open("w", encoding="utf-8") as log:
                subprocess.run(
                    [sys.executable, "-m", module, *arguments],
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        result = json.loads(inspected.read_text(encoding="utf-8"))
        checks = {
            "new_class_registered": result["class_registered"],
            "new_class_predicted": result["top_category"] == fixture["category"],
            "new_article_cited": result["new_article_cited"],
            "citation_contract_passed": result["citation_status"] == "passed",
            "index_ready": result["index"]["status"] == "ready",
            "old_classes_preserved": result["old_category_regression"][
                "all_old_classes_registered"
            ],
            "old_development_accuracy_within_five_points": result["old_category_regression"][
                "accuracy"
            ]
            >= baseline["accuracy"] - 0.05,
        }
        write_json(
            args.output,
            {
                "provenance": fixture["provenance"],
                "database": database,
                "run_directory": str(root),
                "staging": staged,
                "checks": checks,
                "passed": all(checks.values()),
                "inspection": result,
                "old_category_baseline": baseline,
                "active_artifacts_unchanged": active_fingerprint(settings) == before,
                "scope": "Scripted synthetic extension demo, not independent quality evidence. The separate database is retained for inspection.",
            },
        )
        if not all(checks.values()):
            raise RuntimeError("Evolution check failed; inspect the saved report")
    finally:
        if active_fingerprint(settings) != before:
            raise RuntimeError("Evolution unexpectedly changed active artifacts")
    print(f"Evolution verified: {args.output}")


if __name__ == "__main__":
    main()
