import copy

import pytest
from pydantic import ValidationError

from app.ingestion.schema import SupportDocument
from app.ingestion.synthetic import build_dataset, load_scenarios, validate_dataset, write_dataset


def test_dataset_counts_splits_and_causal_pairing():
    scenarios = load_scenarios()
    artifacts = build_dataset(scenarios)
    report = validate_dataset(artifacts)
    assert report["counts"] == dict(
        tickets=510, knowledge_base=60, train=240, dev=120, test=120, documents=330
    )
    assert report["families_per_split"] == dict(train=30, dev=15, test=15)
    lookup = {s["family_id"]: s for s in scenarios}
    for row in artifacts["tickets"]:
        scenario = lookup[row["metadata"]["scenario_family"]]
        if row["outcome_status"] == "simulated_resolved":
            assert scenario["action"] in row["resolution"]
            assert row["metadata"]["diagnostic_findings"] == [scenario["diagnostic"]]
            assert scenario["diagnostic"] not in row["body"]
        else:
            assert row["resolution"] is None and row["outcome_status"] == "unknown"


def test_reproducible_artifacts_and_manifest(tmp_path):
    report = write_dataset(tmp_path)
    assert write_dataset(tmp_path) == report
    assert (tmp_path / "processed/documents.jsonl").is_file()
    assert set(report["file_sha256"]) == {
        "tickets.jsonl",
        "knowledge_base.jsonl",
        "train.jsonl",
        "dev.jsonl",
        "test.jsonl",
        "processed/documents.jsonl",
        "challenge_queries.jsonl",
    }
    assert report["counts"]["challenge_queries"] == 15


def test_validator_rejects_leaking_heldout_ticket():
    artifacts = build_dataset(load_scenarios())
    heldout = next(row for row in artifacts["tickets"] if row["metadata"]["split"] == "test")
    artifacts["documents"].append(heldout)
    with pytest.raises(ValueError, match="Held-out"):
        validate_dataset(artifacts)


@pytest.mark.parametrize("mutation", ["real_claim", "missing_provenance", "missing_evidence"])
def test_simulated_outcomes_cannot_masquerade_as_verified(mutation):
    row = copy.deepcopy(build_dataset(load_scenarios())["tickets"][0])
    if mutation == "real_claim":
        row["outcome_status"] = "verified_resolved"
    elif mutation == "missing_provenance":
        row["metadata"]["is_synthetic"] = False
    else:
        row["metadata"].pop("outcome_evidence")
    with pytest.raises(ValidationError):
        SupportDocument.model_validate(row)
