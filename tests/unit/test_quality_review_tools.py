"""Protect evaluation provenance, response evidence and independently supplied ratings."""

import csv
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from app.evaluation.quality import human_review_summary
from app.evaluation.review import DIMENSIONS, export_ratings, import_ratings, validate_case_pack


def example_case():
    """Provide a reviewable response with a cited diagnostic condition."""
    return {
        "id": "R01",
        "name": "Review",
        "query": "My broadband keeps dropping.",
        "turns": [],
        "expected": ["Keep the cause uncertain"],
        "snapshots": [
            {
                "after_reply": 0,
                "response": {
                    "issues": [
                        {
                            "resolution": {
                                "customer_plan": {
                                    "title": "Investigate",
                                    "summary": "Verify first",
                                    "steps": ["Check the reported symptoms [S1]"],
                                    "note": "Agent review",
                                },
                                "clarification_questions": [],
                                "sources": [
                                    {
                                        "doc_id": "kb_one",
                                        "quotes": [
                                            {"field": "condition", "text": "Verify connectivity"}
                                        ],
                                    }
                                ],
                            }
                        }
                    ]
                },
            }
        ],
    }


def sheet(tmp_path):
    """Export the same frozen evidence and blank scores presented to a reviewer."""
    report = {"run_complete": True, "results": [example_case()]}
    path, csv_path = tmp_path / "report.json", tmp_path / "ratings.csv"
    path.write_text(json.dumps(report), encoding="utf-8")
    export_ratings(report, path, csv_path)
    return report, path, csv_path


def change_sheet(path, update):
    """Simulate human edits without changing the response report."""
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    update(rows)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_blank_sheet_does_not_create_human_scores(tmp_path):
    report, path, csv_path = sheet(tmp_path)
    reviewed = import_ratings(report, path, csv_path)
    assert reviewed["results"][0]["manual_scores"] == dict.fromkeys(DIMENSIONS)
    assert human_review_summary(reviewed["results"])["fully_reviewed_cases"] == 0
    assert "kb_one" in csv_path.read_text() and "Verify connectivity" in csv_path.read_text()


def test_supplied_zero_scores_are_counted_and_original_is_preserved(tmp_path):
    report, path, csv_path = sheet(tmp_path)
    before = deepcopy(report)
    change_sheet(
        csv_path,
        lambda rows: rows[0].update({**dict.fromkeys(DIMENSIONS, "0"), "reviewer": "reviewer-a"}),
    )
    reviewed = import_ratings(report, path, csv_path)
    summary = human_review_summary(reviewed["results"])
    assert summary["fully_reviewed_cases"] == 1
    assert summary["dimensions"]["clarity"]["mean_out_of_2"] == 0
    assert report == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("context", "3"),
        ("context", "true"),
        ("context", "1.5"),
        ("report_sha256", "other"),
        ("id", "unknown"),
    ],
)
def test_invalid_ratings_are_rejected(tmp_path, field, value):
    report, path, csv_path = sheet(tmp_path)
    change_sheet(csv_path, lambda rows: rows[0].update({field: value, "reviewer": "r"}))
    with pytest.raises(ValueError):
        import_ratings(report, path, csv_path)


def test_scores_require_a_reviewer_and_cannot_be_applied_to_changed_report(tmp_path):
    report, path, csv_path = sheet(tmp_path)
    change_sheet(csv_path, lambda rows: rows[0].update({"clarity": "2"}))
    with pytest.raises(ValueError, match="reviewer"):
        import_ratings(report, path, csv_path)
    change_sheet(csv_path, lambda rows: rows[0].update({"reviewer": "r"}))
    path.write_text(json.dumps({**report, "changed": True}))
    with pytest.raises(ValueError, match="exact report"):
        import_ratings(report, path, csv_path)


@pytest.mark.parametrize("duplicate", [False, True])
def test_missing_or_duplicate_rating_rows_are_rejected(tmp_path, duplicate):
    report, path, csv_path = sheet(tmp_path)
    change_sheet(csv_path, lambda rows: rows.append(rows[0]) if duplicate else rows.clear())
    with pytest.raises(ValueError):
        import_ratings(report, path, csv_path)


def test_review_export_preserves_existing_file_and_escapes_formulas(tmp_path):
    report, path, csv_path = sheet(tmp_path)
    with pytest.raises(FileExistsError):
        export_ratings(report, path, csv_path)
    report["results"][0]["query"] = '=HYPERLINK("example")'
    new = tmp_path / "safe.csv"
    export_ratings(report, path, new)
    with new.open(newline="") as stream:
        assert next(csv.DictReader(stream))["complaint"].startswith("'=")


def test_holdout_rejects_split_and_augmentation_leakage(tmp_path):
    for split in ("train", "dev", "test"):
        (tmp_path / f"{split}.jsonl").write_text(json.dumps({"query": split + " complaint"}) + "\n")
    (tmp_path / "training_paraphrases.json").write_text(
        json.dumps({"families": {"a": ["training paraphrase"]}})
    )
    settings = SimpleNamespace(corpus_dir=tmp_path)
    case = example_case()
    pack = {"purpose": "Evaluation", "provenance": "Authored", "cases": [case]}
    validate_case_pack(pack, settings, holdout=True)
    for text in ("train complaint", "DEV complaint", "test complaint", "training paraphrase"):
        case["query"] = text
        with pytest.raises(ValueError, match="overlap"):
            validate_case_pack(pack, settings, holdout=True)


def test_case_pack_rejects_duplicate_ids_and_empty_expectations():
    pack = {
        "purpose": "Evaluation",
        "provenance": "Authored",
        "cases": [example_case(), example_case()],
    }
    with pytest.raises(ValueError, match="unique"):
        validate_case_pack(pack, None)
    pack["cases"] = [example_case()]
    pack["cases"][0]["expected"] = []
    with pytest.raises(ValueError, match="expectations"):
        validate_case_pack(pack, None)
