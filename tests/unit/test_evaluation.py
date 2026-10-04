"""Protect evaluation denominators, immutable reports and family-separated holdouts."""

import json
from pathlib import Path

import numpy as np
import pytest

from app.evaluation.metrics import routing_metrics
from app.evaluation.preparation import freeze_run, validate_holdout
from scripts.evaluate_resolution_path import audit_examples, loss_stage


def test_understanding_fewshots_are_training_families_only():
    audit = audit_examples(Path(__file__).resolve().parents[2] / "data/synthetic/telecom_v1")
    assert len(audit["examples"]) == 30 and audit["heldout_example_count"] == 0
    assert not {"BB03", "SP03", "WF03", "VC03", "NP03", "SM04"}.intersection(
        row["family"] for row in audit["examples"]
    )


def test_stage_trace_distinguishes_applicability_from_final_ranking_loss():
    trace = {
        "scope_status": "supported",
        "supported_scopes": ["billing"],
        "retrieved_kb_ids": ["right", "wrong"],
        "eligibility": [{"doc_id": "right", "rejection": "conflicting_observation"}],
        "selected_pool_ids": ["wrong"],
        "final_source_ids": ["wrong"],
    }
    assert loss_stage(trace, {"right"}) == "applicability"
    trace["eligibility"][0]["rejection"] = None
    assert loss_stage(trace, {"right"}) == "procedure_selection"
    trace["selected_pool_ids"].append("right")
    assert loss_stage(trace, {"right"}) == "ranking_or_source_cap"


def row(family, query, intent="a"):
    return {"scenario_family": family, "query": query, "labels": {"intent": intent}}


def test_zero_acceptance_has_no_accuracy_claim():
    result = routing_metrics(
        [row("a", "query")], np.array([[0.4, 0.35, 0.25]]), ["a", "b", "c"], 0.45, 0.1
    )
    assert result == {"accepted": 0, "total": 1, "coverage": 0.0, "accepted_accuracy": None}


def test_selective_accuracy_uses_only_accepted_queries():
    rows = [row("x", "one"), row("y", "two", "b")]
    result = routing_metrics(
        rows, np.array([[0.8, 0.1, 0.1], [0.4, 0.35, 0.25]]), ["a", "b", "c"], 0.45, 0.1
    )
    assert result["coverage"] == 0.5 and result["accepted_accuracy"] == 1


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


@pytest.mark.parametrize(
    "leak", ["training_family", "development_family", "training_text", "development_text"]
)
def test_test_queries_cannot_overlap_training_or_development(leak):
    train = [row("train", "train query")]
    dev = [row("dev", "dev query")]
    test = [row("test", "test query")]
    if leak == "training_family":
        test[0]["scenario_family"] = "train"
    elif leak == "development_family":
        test[0]["scenario_family"] = "dev"
    elif leak == "training_text":
        test[0]["query"] = " TRAIN QUERY "
    else:
        test[0]["query"] = "DEV QUERY"
    with pytest.raises(ValueError):
        validate_holdout(test, train, dev, "test")


def test_disjoint_holdout_is_accepted():
    validate_holdout([row("test", "test")], [row("train", "train")], [row("dev", "dev")], "test")


def test_routing_metrics_exclude_incompatible_services():
    rows = [row("x", "one"), row("y", "two")]
    scores = np.array([[0.9, 0.05, 0.05], [0.9, 0.05, 0.05]])
    result = routing_metrics(rows, scores, ["a", "b", "c"], 0.4, 0.1, eligible=[True, False])
    assert result["accepted"] == 1 and result["coverage"] == 0.5
