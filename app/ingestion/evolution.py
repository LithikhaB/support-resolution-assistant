"""Prepare isolated corpus and label additions for an executable evolution demonstration."""

import json
from pathlib import Path

from app.ingestion.artifacts import atomic_write, file_sha256, write_json
from app.ingestion.updates import stage_update
from app.understanding.routing import load_category_products
from app.understanding.training import load_split, validate_separation


def prepare_evolution(settings, fixture, output: Path):
    """Preserve the original corpus and split files while adding distinct demonstration families."""
    if output.exists():
        raise ValueError("Choose a new evolution output directory")
    category = fixture["category"]
    groups = fixture["train_families"], fixture["dev_families"]
    if set(groups[0]) & set(groups[1]):
        raise ValueError("Demonstration training and development families overlap")
    prepared = {}
    manifest = json.loads((settings.corpus_dir / "quality_report.json").read_text(encoding="utf-8"))
    for split, families in zip(("train", "dev"), groups, strict=True):
        original = settings.corpus_dir / f"{split}.jsonl"
        rows = load_split(settings.corpus_dir, split)
        if category in {row["labels"]["intent"] for row in rows}:
            raise ValueError("Demonstration category must be new")
        if set(families) & {row["scenario_family"] for row in rows}:
            raise ValueError("Demonstration family conflicts with existing data")
        rows.extend(
            {
                "query_id": f"demo_{family}_{index}",
                "query": text,
                "split": split,
                "scenario_family": family,
                "labels": {"intent": category},
            }
            for family, texts in families.items()
            for index, text in enumerate(texts)
        )
        prepared[split] = rows
    validate_separation(prepared["train"], prepared["dev"])
    output.mkdir(parents=True, exist_ok=False)
    corpus = output / "corpus"
    corpus.mkdir()
    for split, rows in prepared.items():
        original = settings.corpus_dir / f"{split}.jsonl"
        path = corpus / original.name
        atomic_write(path, (json.dumps(row) + "\n" for row in rows))
        manifest["file_sha256"][path.name] = file_sha256(path)
    write_json(corpus / "quality_report.json", manifest)
    augmentation = settings.corpus_dir / "training_paraphrases.json"
    if augmentation.exists():
        (corpus / augmentation.name).write_bytes(augmentation.read_bytes())
    incoming = output / "incoming.jsonl"
    atomic_write(incoming, [json.dumps(fixture["knowledge"]) + "\n"])
    staged = stage_update(
        settings.processed_dir / "documents.jsonl", incoming, corpus / "processed"
    )
    mapping = {
        key: sorted(values)
        for key, values in load_category_products(settings.category_products_path).items()
    }
    mapping[category] = fixture["products"]
    write_json(output / "category_products.json", mapping)
    return staged
