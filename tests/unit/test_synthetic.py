import pytest

from app.ingestion.synthetic import build_dataset, load_scenarios, validate_dataset


def test_validator_rejects_leaking_heldout_ticket():
    artifacts = build_dataset(load_scenarios())
    heldout = next(row for row in artifacts["tickets"] if row["metadata"]["split"] == "test")
    artifacts["documents"].append(heldout)
    with pytest.raises(ValueError, match="Held-out"):
        validate_dataset(artifacts)

    # V3 preserves split files and indexes only training histories. Procedure/ticket
    # steps must match exactly: a historical citation cannot be a guessed link.
    import json
    from pathlib import Path

    root = Path("data/synthetic")
    for name in ("train.jsonl", "dev.jsonl", "test.jsonl"):
        assert (root / "telecom_v2" / name).read_bytes() == (
            root / "telecom_v3" / name
        ).read_bytes()
    rows = [
        json.loads(line)
        for line in (root / "telecom_v3/processed/documents.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    articles = {r["doc_id"]: r for r in rows if r["doc_type"] == "knowledge_base"}
    assert len(articles) == 60
    for name in ("train.jsonl", "dev.jsonl", "test.jsonl"):
        assert (root / "telecom_v3" / name).read_bytes() == (
            root / "telecom_v3_1" / name
        ).read_bytes()
    baselines = [
        json.loads(line)
        for line in (root / "telecom_v3_1/knowledge_base.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(baselines) == 62
    assert {r["metadata"].get("baseline_for") for r in baselines} == {
        None,
        "broadband_outage",
        "slow_broadband",
    }
    from app.resolution.applicability import applicability_issue, evidence_query
    from app.resolution.drafting import draft_resolution
    from app.resolution.evidence import parse_procedure
    from tests.unit.test_resolution import analysis, kb

    for category, complaint in (
        ("broadband_outage", "Oh brilliant, another lovely evening of no internet."),
        (
            "slow_broadband",
            "Internet romba slow ah irukku since morning, router restart pannitten no use.",
        ),
    ):
        observed = analysis(complaint)
        observed.category, observed.category_basis = category, "explicit_report"
        article_row = next(r for r in baselines if r["metadata"].get("baseline_for") == category)
        evidence = kb().model_copy(
            update={
                "doc_id": article_row["doc_id"],
                "content": article_row["body"],
                "metadata": article_row["metadata"],
                "title": article_row["title"],
            }
        )
        assert applicability_issue(parse_procedure(evidence), observed) is None
        assert article_row["title"] in evidence_query(complaint, observed)
        result = draft_resolution(observed, [evidence])
        assert len(result.sources) == 1
        assert len(result.customer_plan.steps) >= 5
    for row in rows:
        if row["doc_type"] == "knowledge_base":
            assert len(row["metadata"]["procedure"]["steps"]) == 5
            assert "expert review" in row["metadata"]["authorship"]
        else:
            assert row["metadata"]["split"] == "train"
        if row["doc_type"] == "resolved_ticket":
            assert row["outcome_status"] == "simulated_resolved"
            assert row["metadata"]["ticket_id"] == row["doc_id"]
            article = articles[row["metadata"]["kb_refs"][0]]
            for step in article["metadata"]["procedure"]["steps"]:
                assert f"Step {step['id']}: {step['text']}\n" in row["resolution"]
                assert ".." not in step["text"]
