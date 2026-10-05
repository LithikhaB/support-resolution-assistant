"""Protect evaluation denominators, immutable reports and family-separated holdouts."""

import json
from pathlib import Path

import pytest

from app.evaluation.preparation import freeze_run
from scripts.evaluate_resolution_path import audit_examples


def test_understanding_fewshots_are_training_families_only():
    audit = audit_examples(Path(__file__).resolve().parents[2] / "data/synthetic/telecom_v1")
    assert len(audit["examples"]) == 30 and audit["heldout_example_count"] == 0
    assert not {"BB03", "SP03", "WF03", "VC03", "NP03", "SM04"}.intersection(
        row["family"] for row in audit["examples"]
    )


def row(family, query, intent="a"):
    return {"scenario_family": family, "query": query, "labels": {"intent": intent}}


def test_freeze_prevents_overwrite_and_changed_policy(tmp_path):
    output = tmp_path / "report.json"
    snapshot = {"split": "test", "model": "frozen"}
    freeze = freeze_run(output, snapshot)
    assert json.loads(freeze.read_text())["fingerprint"] == snapshot
    assert freeze_run(output, snapshot) == freeze
    with pytest.raises(ValueError, match="Inputs changed"):
        freeze_run(output, {**snapshot, "model": "changed"})
    output.write_text("{}")
    with pytest.raises(ValueError, match="already exists"):
        freeze_run(output, snapshot)
