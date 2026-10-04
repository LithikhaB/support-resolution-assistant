"""Report per-area executable contract checks separately from unscored manual expectations."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from app.ingestion.artifacts import file_sha256, write_json


def summarize(report):
    """A structural pass is not an assertion that the natural-language expectations passed."""
    areas = defaultdict(Counter)
    failures = []
    providers = Counter()
    categories_unknown = sentiments_unknown = 0
    for case in report["results"]:
        area = areas[case.get("area", "unspecified")]
        area["cases"] += 1
        case_passed = True
        for snapshot in case["snapshots"]:
            area["snapshots"] += 1
            if "error" in snapshot:
                case_passed = False
                failures.append(
                    {
                        "case": case["id"],
                        "after_reply": snapshot["after_reply"],
                        "error": snapshot["error"],
                    }
                )
                continue
            issues = snapshot["response"]["issues"]
            good = bool(issues)
            for issue in issues:
                result = issue["resolution"]
                plan = result.get("language_plan") or result["customer_plan"]
                good &= (
                    result["validation"]["status"] == "passed"
                    and result["agent_review_required"] is True
                    and result["decision"]["handoff_created"] is False
                    and bool(plan and plan["steps"])
                    and bool(result["decision"]["action"])
                )
                providers[result["language_status"]] += 1
                categories_unknown += result["analysis"]["category"] is None
                sentiments_unknown += result["analysis"]["sentiment"]["value"] == "unknown"
            area["contract_passed_snapshots"] += good
            case_passed &= good
            if not good:
                failures.append(
                    {
                        "case": case["id"],
                        "after_reply": snapshot["after_reply"],
                        "error": "response_contract_failed",
                    }
                )
        area["contract_passed_cases"] += case_passed
    return {
        "run_complete": report["run_complete"],
        "measurement": "HTTP success, nonempty plan, passed exact-source validation, agent review required and no handoff claim; not relevance, diagnosis or manual expectation accuracy",
        "per_area": {
            name: {
                **count,
                "contract_case_pass_rate": count["contract_passed_cases"] / count["cases"],
            }
            for name, count in sorted(areas.items())
        },
        "language_statuses": dict(providers),
        "unknown_category_issue_snapshots": categories_unknown,
        "unknown_sentiment_issue_snapshots": sentiments_unknown,
        "failures": failures,
        "independent_human_ratings": 0,
        "manual_expectation_cases_unscored": len(report["results"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new filename; existing reports are immutable")
    result = summarize(json.loads(args.report.read_text(encoding="utf-8")))
    result["report_sha256"] = file_sha256(args.report)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
