"""Protect evaluation provenance, response evidence and independently supplied ratings."""

import csv
import json

import pytest

from app.evaluation.quality import human_review_summary
from app.evaluation.review import DIMENSIONS, export_ratings, import_ratings


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


def test_scores_require_a_reviewer_and_cannot_be_applied_to_changed_report(tmp_path):
    report, path, csv_path = sheet(tmp_path)
    change_sheet(csv_path, lambda rows: rows[0].update({"clarity": "2"}))
    with pytest.raises(ValueError, match="reviewer"):
        import_ratings(report, path, csv_path)
    change_sheet(csv_path, lambda rows: rows[0].update({"reviewer": "r"}))
    path.write_text(json.dumps({**report, "changed": True}))
    with pytest.raises(ValueError, match="exact report"):
        import_ratings(report, path, csv_path)
