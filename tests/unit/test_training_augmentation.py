"""Protect training provenance and keep paraphrases out of evaluation families."""

import json

import pytest

from app.understanding.augmentation import augment_training
from app.understanding.training import validate_separation


def training_row():
    """Provide a training family whose labels may be inherited, never invented."""
    return {
        "query_id": "original",
        "scenario_family": "train_a",
        "query": "original complaint",
        "labels": {"intent": "billing"},
    }


def write_pack(path, families):
    """Save the same declarative input accepted by the training command."""
    path.write_text(
        json.dumps({"provenance": "test fixture", "families": families}), encoding="utf-8"
    )


def test_paraphrases_inherit_only_intent_and_do_not_mutate_training_rows(tmp_path):
    original = training_row()
    path = tmp_path / "extra.json"
    write_pack(path, {"train_a": ["new wording"]})
    rows, checksum = augment_training([original], path)
    assert len(rows) == 2 and len(checksum) == 64
    assert rows[-1]["labels"] == {"intent": "billing"}
    assert rows[-1]["split"] == "train"
    assert original == training_row()


@pytest.mark.parametrize(
    "families",
    [
        {"dev_a": ["unseen family"]},
        {"train_a": [" ORIGINAL COMPLAINT "]},
        {"train_a": ["same", "Same"]},
        {"train_a": [""]},
        {"train_a": [None]},
        {"train_a": "not a list"},
        {},
    ],
)
def test_invalid_or_leaking_paraphrases_are_rejected(tmp_path, families):
    path = tmp_path / "extra.json"
    write_pack(path, families)
    with pytest.raises(ValueError):
        augment_training([training_row()], path)


def test_development_text_cannot_be_smuggled_under_a_training_family(tmp_path):
    path = tmp_path / "extra.json"
    write_pack(path, {"train_a": ["development complaint"]})
    rows, _ = augment_training([training_row()], path)
    dev = [{**training_row(), "scenario_family": "dev_a", "query": "development complaint"}]
    with pytest.raises(ValueError, match="Complaints overlap"):
        validate_separation(rows, dev)


def test_missing_optional_paraphrases_preserve_existing_training(tmp_path):
    rows = [training_row()]
    assert augment_training(rows, tmp_path / "missing.json") == (rows, None)
