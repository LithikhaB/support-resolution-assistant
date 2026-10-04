"""Export blank human review sheets or import completed ratings into a new report."""

import argparse
import json
from pathlib import Path

from app.evaluation.quality import human_review_summary
from app.evaluation.review import export_ratings, import_ratings
from app.ingestion.artifacts import write_json


def main():
    """Preserve original response evidence and never assign automatic human scores."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--export", type=Path, help="New CSV with blank 0–2 ratings")
    mode.add_argument("--ratings", type=Path, help="Completed CSV tied to this exact report")
    parser.add_argument("--output", type=Path, help="New reviewed JSON; required with --ratings")
    args = parser.parse_args()
    if args.ratings and (args.output is None or args.output.exists()):
        parser.error("--ratings requires a new --output path")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("run_complete") is not True:
        parser.error("review a completed run")
    if args.export:
        export_ratings(report, args.report, args.export)
        print(f"Blank human review sheet: {args.export}")
    else:
        reviewed = import_ratings(report, args.report, args.ratings)
        reviewed["human_review"] = human_review_summary(reviewed["results"])
        write_json(args.output, reviewed)
        print(json.dumps(reviewed["human_review"], indent=2))


if __name__ == "__main__":
    main()
