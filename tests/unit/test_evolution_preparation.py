"""Verify that new categories are staged without modifying active files or leaking families."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.ingestion.artifacts import file_sha256
from app.ingestion.evolution import prepare_evolution
from app.understanding.training import load_split


def setup(tmp_path):
    """Build checksum-valid train/development fixtures and a minimal active KB."""
    corpus = tmp_path / "active"
    processed = corpus / "processed"
    processed.mkdir(parents=True)
    files = {}
    for split in ("train", "dev"):
        rows = [
            {
                "query_id": split,
                "query": split + " existing complaint",
                "split": split,
                "scenario_family": split + "_old",
                "labels": {"intent": "old_category"},
            }
        ]
        path = corpus / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        files[path.name] = file_sha256(path)
    (corpus / "quality_report.json").write_text(json.dumps({"file_sha256": files}))
    (processed / "documents.jsonl").write_text(
        json.dumps(
            {
                "doc_id": "existing",
                "doc_type": "knowledge_base",
                "title": "Existing",
                "body": "Existing conditional support procedure.",
            }
        )
        + "\n"
    )
    settings = SimpleNamespace(
        corpus_dir=corpus,
        processed_dir=processed,
        category_products_path=tmp_path / "absent_mapping.json",
    )
    fixture = json.loads(Path("data/evolution/dns_category_demo.json").read_text())
    return settings, fixture


def test_new_class_files_are_trainable_and_preserve_active_artifacts(tmp_path):
    settings, fixture = setup(tmp_path)
    paths = list(settings.corpus_dir.rglob("*"))
    before = {str(path): path.read_bytes() for path in paths if path.is_file()}
    output = tmp_path / "demo"
    staged = prepare_evolution(settings, fixture, output)
    assert staged["added"] == 1 and staged["documents"] == 2
    assert fixture["category"] in {
        row["labels"]["intent"] for row in load_split(output / "corpus", "train")
    }
    assert fixture["category"] in json.loads((output / "category_products.json").read_text())
    assert all(Path(path).read_bytes() == data for path, data in before.items())


def test_cross_split_family_leakage_is_rejected_before_writes(tmp_path):
    settings, fixture = setup(tmp_path)
    fixture["train_families"] = {"dev_old": ["new training query"]}
    output = tmp_path / "demo"
    with pytest.raises(ValueError, match="overlap"):
        prepare_evolution(settings, fixture, output)
    assert not output.exists()


def test_evolution_refuses_unverified_source_splits_and_existing_output(tmp_path):
    settings, fixture = setup(tmp_path)
    before = deepcopy(fixture)
    (settings.corpus_dir / "train.jsonl").write_text("changed source")
    output = tmp_path / "demo"
    with pytest.raises(ValueError, match="checksum"):
        prepare_evolution(settings, fixture, output)
    assert fixture == before and not output.exists()
    output.mkdir()
    with pytest.raises(ValueError, match="new evolution"):
        prepare_evolution(settings, fixture, output)
