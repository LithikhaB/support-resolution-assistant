"""Record natural-language CLI conversations for manual response-quality review."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import sleep

from fastapi import HTTPException
from pydantic import ValidationError

from app.api.resolution import run_request
from app.ingestion.artifacts import digest, write_json
from app.resolution.conversation import ConversationRequest, resolve_conversation
from app.resolution.service import get_resolution_service


def evaluate_case(case, service, delay_seconds=0):
    """Replay every prefix to expose how the plan and clarification evolve after each reply."""
    turns = case["turns"]
    snapshots = []
    for count in range(len(turns) + 1):
        if delay_seconds:
            sleep(delay_seconds)
        try:
            request = ConversationRequest(query=case["query"], turns=turns[:count])
            response = run_request(lambda: resolve_conversation(request, service))
            snapshots.append({"after_reply": count, "response": response.model_dump(mode="json")})
        except HTTPException as exc:
            snapshots.append({"after_reply": count, "error": exc.detail, "status": exc.status_code})
        except ValidationError as exc:
            snapshots.append(
                {
                    "after_reply": count,
                    "error": "; ".join(error["msg"] for error in exc.errors()),
                    "status": 422,
                }
            )
    return {
        **case,
        "snapshots": snapshots,
        "manual_scores": {
            "context": None,
            "relevance": None,
            "conversation": None,
            "grounding": None,
            "clarity": None,
        },
        "review_notes": "",
    }


def print_case(case):
    """Show the same plan used by the UI together with the actual remaining questions."""
    print(f"\n{case['id']} | {case['name']}\nComplaint: {case['query']}")
    for snapshot in case["snapshots"]:
        count = snapshot["after_reply"]
        print(f"\nAfter reply {count}")
        if count:
            print("Customer: " + case["turns"][count - 1]["message"])
        if "error" in snapshot:
            print("ERROR: " + str(snapshot["error"]))
            continue
        for issue in snapshot["response"]["issues"]:
            result = issue["resolution"]
            print(
                f"Issue {issue['issue_id']} | Category: {result['analysis']['category']} | {result['decision']['action']}"
            )
            plan = result.get("language_plan") or result["customer_plan"]
            if plan:
                print(plan["title"] + "\n" + (result.get("language_summary") or plan["summary"]))
                for index, step in enumerate(plan["steps"], 1):
                    print(f"  {index}. {step}")
                print(plan["note"])
            print("Remaining questions:")
            for question in result["clarification_questions"]:
                print("  - " + question)
            if not result["clarification_questions"]:
                print("  None")
    print("\nReview against these expectations:")
    for expected in case["expected"]:
        print("  - " + expected)


def main():
    """Save a uniquely named review artifact without replacing frozen evaluations."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", type=Path, default=Path("data/evaluation/response_quality_cases.json")
    )
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0,
        help="Pause between snapshots for provider limits (0 to 60).",
    )
    args = parser.parse_args()
    if not 0 <= args.delay_seconds <= 60:
        parser.error("delay-seconds must be between 0 and 60")
    now = datetime.now(timezone.utc)
    output = args.output or Path("data/evaluation/quality_runs") / (
        now.strftime("%Y%m%dT%H%M%S%fZ") + ".json"
    )
    if output.exists():
        parser.error("choose a new output file; existing reports are preserved")
    try:
        pack = json.loads(args.cases.read_text(encoding="utf-8"))
        available = {case["id"] for case in pack["cases"]}
        if set(args.case) - available:
            parser.error("unknown case ID; available: " + ", ".join(sorted(available)))
        selected = [case for case in pack["cases"] if not args.case or case["id"] in args.case]
        for case in selected:
            if not isinstance(case["query"], str) or not isinstance(case["turns"], list):
                raise ValueError("each case needs a query string and turns list")
    except (OSError, ValueError, KeyError, TypeError, ValidationError) as exc:
        parser.exit(2, f"Invalid evaluation input: {exc}\n")
    report = {
        "created_at": now.isoformat(),
        "cases_sha256": digest(pack),
        "provenance": pack["provenance"],
        "purpose": pack["purpose"],
        "scoring": pack["scoring"],
        "results": [],
    }
    service = get_resolution_service()
    for case in selected:
        result = evaluate_case(case, service, args.delay_seconds)
        report["results"].append(result)
        write_json(output, report)
        print_case(result)
    print(f"\nSaved full evidence and empty manual scoring fields: {output}")
    print("No automatic quality score was assigned. Citation integrity is not response quality.")


if __name__ == "__main__":
    main()
