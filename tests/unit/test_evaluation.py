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


def test_code_fingerprint_preserves_file_scope_and_platform_path_keys(tmp_path):
    import os

    from app.ingestion.artifacts import code_fingerprint

    fixtures = {
        "app/nested/model.py": b"class Model: pass\n",
        "app/routes.py": b"API = 1\n",
        "scripts/cli.py": b"CLI = 1\n",
        "scripts/nested/ignored.py": b"not a top-level command\n",
        "README.md": b"documentation\n",
        ".env": b"private configuration\n",
    }
    for name, content in fixtures.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    expected = (
        "a3dd7f8f9b06fe71c316d10992b980d2fb208f32509377372e8d6b2f4b9a0f59"
        if os.name == "nt"
        else "902663224b81120084a337a6ab29150a0c9db35a26ce904246006c9e6241bbdf"
    )
    assert code_fingerprint(tmp_path) == expected
    (tmp_path / ".env").write_bytes(b"changed private configuration\n")
    assert code_fingerprint(tmp_path) == expected
    (tmp_path / "app/routes.py").write_bytes(b"API = 2\n")
    assert code_fingerprint(tmp_path) != expected
