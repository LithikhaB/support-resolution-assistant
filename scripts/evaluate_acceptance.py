"""Evaluate authored conversation acceptance cases without changing frozen test reports."""

import argparse
from statistics import median
from time import perf_counter

from app.config.settings import get_settings
from app.evaluation.preparation import fingerprint
from app.ingestion.artifacts import digest, write_json
from app.resolution.conversation import ConversationRequest, resolve_conversation
from app.resolution.service import get_resolution_service

CASES = [
    {
        "name": "wireless_after_wired_test",
        "query": "My home broadband disconnects during meetings. I already restarted the router.",
        "turns": [{"issue_id": 1, "message": "Ethernet works. All wireless devices disconnect."}],
        "check": "wireless",
    },
    {
        "name": "settled_duplicate_payment",
        "query": "My invoice includes a duplicate payment.",
        "check": "billing",
    },
    {
        "name": "separate_billing_and_connection",
        "query": "My broadband is unavailable. Also my bill includes an extra charge.",
        "check": "two_issues",
    },
    {
        "name": "reported_neighbourhood_outage",
        "query": "Several neighbours lost broadband at the same time.",
        "check": "urgent",
    },
    {
        "name": "unrelated_writing_request",
        "query": "Write a poem about summer.",
        "check": "unsupported",
    },
    {
        "name": "corrected_wired_observation",
        "query": "My broadband drops.",
        "turns": [
            {"issue_id": 1, "message": "Ethernet works."},
            {"issue_id": 1, "message": "Correction: Ethernet drops."},
        ],
        "check": "correction",
    },
    {
        "name": "reported_recovery",
        "query": "My broadband drops.",
        "turns": [{"issue_id": 1, "observations": {"impact": "working"}}],
        "check": "recovery",
    },
    {
        "name": "answered_mobile_question",
        "query": "My mobile service has a problem.",
        "turns": [
            {"issue_id": 1, "observations": {"mobile_services": "texts", "impact": "complete_loss"}}
        ],
        "check": "mobile",
    },
]


def check_response(kind, result):
    """Evaluate explicit behavioural criteria independently of category score confidence."""
    response = result.issues[0].resolution
    if kind == "wireless":
        return (
            bool(response.sources)
            and all(
                q.text == "home_wifi"
                for s in response.sources
                for q in s.quotes
                if q.field == "scope"
            )
            and bool(response.acknowledged_actions)
        )
    if kind == "billing":
        return any("charge or payment" in q for q in response.clarification_questions) and not any(
            "devices or people" in q for q in response.clarification_questions
        )
    if kind == "two_issues":
        return len(result.issues) == 2 and all(i.resolution.sources for i in result.issues)
    if kind == "urgent":
        return response.decision.priority == "urgent" and response.decision.action == "escalate"
    if kind == "unsupported":
        return response.status == "unsupported_request" and not response.sources
    if kind == "correction":
        return {
            f.value for f in response.analysis.reported_facts if f.name == "wired_connection"
        } == {"failing"}
    if kind == "recovery":
        return (
            "customer reports recovery" in response.draft.lower()
            and "procedure proposes" not in response.draft
        )
    if kind == "mobile":
        return response.analysis.severity.value == "high" and not any(
            "Are calls" in q for q in response.clarification_questions
        )
    raise ValueError("Unknown acceptance criterion")


def main():
    """Save inspectable results and separate cold-start timing from subsequent requests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/evaluation/acceptance_workflow.json")
    args = parser.parse_args()
    from pathlib import Path

    output = Path(args.output)
    if output.exists():
        parser.error("Output already exists; choose a new --output to preserve earlier results.")
    settings = get_settings()
    before = fingerprint(settings, "dev")
    service = get_resolution_service()
    runs = []
    for case in CASES:
        started = perf_counter()
        try:
            request = ConversationRequest(query=case["query"], turns=case.get("turns", []))
            response = resolve_conversation(request, service)
            valid = all(i.resolution.validation.status == "passed" for i in response.issues)
            runs.append(
                {
                    "name": case["name"],
                    "passed": bool(valid and check_response(case["check"], response)),
                    "elapsed_ms": (perf_counter() - started) * 1000,
                    "response": response.model_dump(mode="json"),
                }
            )
        except Exception as exc:
            runs.append(
                {
                    "name": case["name"],
                    "passed": False,
                    "error_type": type(exc).__name__,
                    "elapsed_ms": (perf_counter() - started) * 1000,
                }
            )
        print(f"{case['name']}: {'PASS' if runs[-1]['passed'] else 'FAIL'}", flush=True)
    if fingerprint(settings, "dev") != before:
        raise ValueError("Evaluation inputs changed during the run")
    summary = {
        "passed": sum(r["passed"] for r in runs),
        "total": len(runs),
        "first_request_ms": runs[0]["elapsed_ms"],
        "subsequent_median_ms": median(r["elapsed_ms"] for r in runs[1:]),
    }
    write_json(
        output,
        {
            "purpose": "Author-designed acceptance regression cases; not independent held-out accuracy or a representative performance benchmark.",
            "fingerprint": before,
            "cases_sha256": digest(CASES),
            "cases": CASES,
            "summary": summary,
            "runs": runs,
        },
    )
    print(summary)
    if not all(r["passed"] for r in runs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
