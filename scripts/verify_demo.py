"""Record authored HTTP regressions separately from held-out accuracy and human review."""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from time import sleep
from urllib.parse import urlparse

import httpx

CASES = [
    (
        "evening",
        "My broadband drops every evening around 8 and I've already restarted the router twice, I work from home and this is costing me.",
        "intermittent_broadband",
        "frustrated",
        "high",
    ),
    (
        "storm",
        "Since the storm on Tuesday my internet cuts out for about a minute every hour or so. The router light stays green. I unplugged it for ten minutes, twice. I run a small online tailoring shop so it's costing me orders.",
        "intermittent_broadband",
        "frustrated",
        "high",
    ),
    (
        "optical",
        "No internet at all. I restarted the router three times, checked every cable, reset it to factory settings and waited an hour. The optical box has a red light.",
        "broadband_outage",
        None,
        "high",
    ),
    (
        "duplicate_payment",
        "I paid the same broadband invoice twice. Both payments show settled in the provider portal, but my service works. Please help me dispute the duplicate payment.",
        "billing_dispute",
        None,
        None,
    ),
    (
        "wifi",
        "My Wi-Fi keeps disconnecting on every wireless device, but Ethernet stays working. I already restarted the router twice.",
        "wifi_connectivity",
        None,
        "medium",
    ),
    (
        "shared_outage",
        "Our whole street has no internet and several neighbours lost broadband at once. Please advise.",
        None,
        None,
        "critical",
    ),
    (
        "tanglish",
        "Internet romba slow ah irukku since morning, router restart pannitten no use.",
        "slow_broadband",
        "frustrated",
        "medium",
    ),
    ("irrelevant", "What is photosynthesis?", None, None, None),
]


def verify(result, name, category, sentiment, severity):
    """Check explicit regressions and citation integrity, never assign a human quality score."""
    analysis = result["analysis"]
    plan = result.get("language_plan") or result["customer_plan"]
    if name == "irrelevant":
        assert result["status"] == "unsupported_request" and not plan["steps"]
        return
    assert result["validation"]["status"] == "passed"
    assert result["agent_review_required"] and not result["decision"]["handoff_created"]
    assert result["sources"] and plan["steps"]
    if category:
        assert analysis["category"] == category, analysis["category"]
    if sentiment:
        assert analysis["sentiment"]["value"] == sentiment
    if severity:
        assert analysis["severity"]["value"] == severity
    allowed = {s["citation_id"] for s in result["sources"] + result["historical_cases"]}
    assert set(re.findall(r"\[([ST]\d+)\]", " ".join(plan["steps"]))) <= allowed
    if name in {"storm", "optical", "tanglish", "wifi"}:
        assert not any(
            "congest" in s["title"].lower() or "backhaul" in s["title"].lower()
            for s in result["sources"]
        )
    if name == "storm":
        assert any("ten minutes, twice" in item for item in result["acknowledged_actions"])
    if name == "optical":
        assert [s["doc_id"] for s in result["sources"]] == ["syn_kb_BB01"]
    if name == "duplicate_payment":
        assert [s["doc_id"] for s in result["sources"]] == ["syn_kb_BD01"]
    if name == "shared_outage":
        assert result["decision"]["action"] == "escalate"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()
    url = urlparse(args.url)
    if (
        url.hostname not in {"127.0.0.1", "localhost"}
        or url.scheme != "http"
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        parser.error("use a plain local http URL")
    if args.output.exists() or set(args.case) - {case[0] for case in CASES}:
        parser.error("choose known case names and a new output path")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Authored regression checks, not held-out or human-rated plan quality",
        "cases": [],
        "complete": False,
    }
    with httpx.Client(base_url=args.url, timeout=180) as client:
        for name, query, category, sentiment, severity in CASES:
            if args.case and name not in args.case:
                continue
            row = {"case": name, "query": query, "passed": False}
            try:
                for attempt in range(3):
                    response = client.post("/api/v1/resolve", json={"query": query})
                    if response.status_code not in {429, 503} or attempt == 2:
                        break
                    sleep(min(5, max(1, int(response.headers.get("Retry-After", 2)))))
                response.raise_for_status()
                row["response"] = response.json()
                verify(row["response"], name, category, sentiment, severity)
                row["passed"] = True
            except (httpx.HTTPError, AssertionError, KeyError, ValueError) as exc:
                row["error_type"] = type(exc).__name__
            report["cases"].append(row)
            print(
                name,
                "passed" if row["passed"] else "FAILED",
                row.get("response", {}).get("language_status"),
                flush=True,
            )
    report["complete"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    if not all(row["passed"] for row in report["cases"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
