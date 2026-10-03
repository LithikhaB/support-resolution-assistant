"""Check recorded golden behaviours and compare operational drift indicators."""

import argparse
import json
from pathlib import Path

from app.evaluation.quality import summarize
from app.ingestion.artifacts import write_json


def main():
    """Preserve previous reports and require matching case packs for rate comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--golden", type=Path, default=Path("data/evaluation/response_golden.json"))
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output file")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    golden = json.loads(args.golden.read_text(encoding="utf-8"))
    result = summarize(report, golden)
    if args.baseline:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        if report["cases_sha256"] != baseline["cases_sha256"] or {
            c["id"] for c in report["results"]
        } != {c["id"] for c in baseline["results"]}:
            parser.error("baseline must use the same case pack and selected cases")
        before = summarize(baseline, golden)
        result["baseline_rate_changes"] = {
            key: result["rates"][key] - before["rates"][key] for key in result["rates"]
        }
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
