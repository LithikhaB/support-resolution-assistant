"""Audit few-shot provenance and measure the actual resolution path with stage traces."""

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import sleep

from app.config.settings import get_settings
from app.ingestion.artifacts import digest, write_json
from app.llm.providers import ProviderChain
from app.resolution.models import ResolutionRequest
from app.resolution.service import ResolutionService
from app.understanding.classifier import CategoryClassifier
from app.understanding.llm_classifier import CATEGORY_EXAMPLES
from app.understanding.models import AnalyzeRequest
from app.understanding.service import UnderstandingService
from app.understanding.training import load_split
from scripts.evaluate_response_quality import runtime_fingerprint


def audit_examples(directory):
    """Reject examples drawn from any held-out diagnostic family before remote evaluation."""
    training = {row["scenario_family"] for row in load_split(directory, "train")}
    with (directory / "scenarios.psv").open(encoding="utf-8") as stream:
        scenarios = list(csv.DictReader(stream, delimiter="|"))
    provenance = []
    for category, examples in CATEGORY_EXAMPLES.items():
        for text in examples:
            matches = [row for row in scenarios if text in (row["complaint"], row["paraphrase"])]
            if len(matches) != 1 or matches[0]["family_id"] not in training:
                raise ValueError("Few-shot example is not uniquely traceable to a training family")
            if matches[0]["category"] != category:
                raise ValueError("Few-shot category does not match its training family")
            provenance.append({"category": category, "family": matches[0]["family_id"]})
    return {
        "examples": provenance,
        "prompt_examples_sha256": digest(CATEGORY_EXAMPLES),
        "heldout_example_count": 0,
    }


def loss_stage(trace, expected):
    """Separate scope, retrieval, applicability, selection and final-cap losses."""
    if not trace["supported_scopes"] or trace["scope_status"] == "unsupported":
        return "scope_gate"
    if not expected.intersection(trace["retrieved_kb_ids"]):
        return "retrieval"
    eligible = {row["doc_id"] for row in trace["eligibility"] if row["rejection"] is None}
    if not expected.intersection(eligible):
        return "applicability"
    if not expected.intersection(trace["selected_pool_ids"]):
        return "procedure_selection"
    if not expected.intersection(trace["final_source_ids"]):
        return "ranking_or_source_cap"
    return "retained"


def summarize(records):
    """Keep actual language success, fallback, grounding and human quality distinct."""
    count = len(records)
    generated = [
        r for r in records if r["response"].get("language_status") == "generated_for_review"
    ]
    extracted = [
        r for r in records if r["response"]["analysis"]["language_method"].endswith("extraction_v1")
    ]

    def matches(row, key):
        analysis = row["response"]["analysis"]
        return (
            analysis["category"] == row["labels"]["intent"]
            if key == "category"
            else (analysis[key]["value"] == row["labels"][key])
        )

    return {
        "attempted_cases": count,
        "families": len({r["family"] for r in records}),
        "category_matches_all_attempts": sum(matches(r, "category") for r in records),
        "category_uncertain": sum(r["response"]["analysis"]["category"] is None for r in records),
        "product_family_matches": sum(
            {
                "fibre_broadband": "broadband",
                "mobile_data": "mobile",
                "mobile_roaming": "mobile",
                "mobile_sms": "mobile",
                "mobile_voice": "mobile",
                "sim": "mobile",
            }.get(r["labels"]["product"], r["labels"]["product"])
            in {p["product"] for p in r["response"]["analysis"]["products"]}
            for r in records
        ),
        "language_understanding_successes": len(extracted),
        "language_category_matches_successful_extractions": sum(
            matches(r, "category") for r in extracted
        ),
        "severity_matches": sum(matches(r, "severity") for r in records),
        "severity_unknown": sum(
            r["response"]["analysis"]["severity"]["value"] == "unknown" for r in records
        ),
        "sentiment_matches": sum(matches(r, "sentiment") for r in records),
        "generated_answers": len(generated),
        "generated_citation_contract_passes": sum(
            r["response"]["validation"]["status"] == "passed" for r in generated
        ),
        "generated_model_grounding_passes": sum(
            r["response"]["faithfulness_status"] == "model_checked" for r in generated
        ),
        "expected_kb_retained": sum(r["loss_stage"] == "retained" for r in records),
        "loss_stages": dict(Counter(r["loss_stage"] for r in records)),
        "answers_with_history": sum(bool(r["response"]["historical_cases"]) for r in records),
        "independent_human_ratings": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--understanding-only", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--delay-seconds", type=float, default=0)
    parser.add_argument("--minimum-generated", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.delay_seconds <= 60 or (args.limit is not None and args.limit < 1):
        parser.error("delay must be 0–60 seconds and limit must be positive")
    if args.output.exists() and not args.resume:
        parser.error("choose a new report path or explicitly resume this run")
    settings = get_settings().model_copy(update={"llm_enabled": args.live})
    provenance = audit_examples(settings.corpus_dir)
    rows = sorted(
        load_split(settings.corpus_dir, args.split),
        key=lambda row: (row["query_id"].rsplit("_", 1)[-1], row["scenario_family"]),
    )
    rows = rows[: args.limit] if args.limit else rows
    fingerprint = runtime_fingerprint(settings)
    config = {
        "split": args.split,
        "live": args.live,
        "understanding_only": args.understanding_only,
        "query_ids": [row["query_id"] for row in rows],
        "lexical_backend": settings.lexical_backend,
    }
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fingerprint": fingerprint,
        "config": config,
        "fewshot_audit": provenance,
        "records": [],
        "results": [],
        "run_complete": False,
        "provenance": "Authored synthetic diagnostic-family split; previously observed development data, not independent customer accuracy.",
        "purpose": "Trace the actual resolution path and review grounded answers.",
        "scoring": "Independent reviewer: 0 inadequate, 1 partly adequate, 2 adequate; leave conversation blank for one-turn cases.",
    }
    if args.resume and args.output.exists():
        report = json.loads(args.output.read_text(encoding="utf-8"))
        if report["fingerprint"] != fingerprint or report["config"] != config:
            raise ValueError("Runtime or requested evaluation changed; use a new report")
    chain = ProviderChain(settings)
    understanding = UnderstandingService(
        CategoryClassifier.load(settings.understanding_model_path, settings=settings),
        settings=settings,
        language=chain,
    )
    service = ResolutionService(settings=settings, understanding=understanding, language=chain)
    completed = {r["query_id"] for r in report["records"]}
    for row in rows:
        if row["query_id"] in completed:
            continue
        if args.delay_seconds and report["records"]:
            sleep(args.delay_seconds)
        trace = {}
        if args.understanding_only:
            analysis = understanding.analyze(AnalyzeRequest(query=row["query"]))
            response = {
                "analysis": analysis.model_dump(mode="json"),
                "historical_cases": [],
                "language_status": "not_requested",
            }
            stage = "not_requested"
        else:
            result = service.resolve(ResolutionRequest(query=row["query"]), trace=trace)
            response = result.model_dump(mode="json")
            stage = loss_stage(trace, set(row["relevant_kb_ids"]))
            report["results"].append(
                {
                    "id": row["query_id"],
                    "name": row["scenario_family"],
                    "query": row["query"],
                    "turns": [],
                    "expected": [row["required_diagnostic"], row["forbidden_action"]],
                    "snapshots": [
                        {
                            "after_reply": 0,
                            "response": {
                                "issues": [
                                    {
                                        "issue_id": 1,
                                        "complaint": row["query"],
                                        "resolution": response,
                                    }
                                ]
                            },
                        }
                    ],
                    "manual_scores": {
                        k: None
                        for k in ("context", "relevance", "conversation", "grounding", "clarity")
                    },
                    "review_notes": "",
                }
            )
        report["records"].append(
            {
                "query_id": row["query_id"],
                "family": row["scenario_family"],
                "labels": row["labels"],
                "expected_kb_ids": row["relevant_kb_ids"],
                "trace": trace,
                "loss_stage": stage,
                "response": response,
            }
        )
        report["summary"] = summarize(report["records"])
        write_json(args.output, report)
        print(
            json.dumps(
                {
                    "case": row["query_id"],
                    "category": response["analysis"]["category"],
                    "understanding": response["analysis"]["language_method"],
                    "draft": response["language_status"],
                    "loss_stage": stage,
                }
            ),
            flush=True,
        )
    if runtime_fingerprint(settings) != fingerprint:
        raise ValueError("Runtime changed during evaluation; partial evidence retained")
    report["run_complete"] = True
    report["minimum_generated_target_met"] = (
        report["summary"]["generated_answers"] >= args.minimum_generated
    )
    write_json(args.output, report)
    print(json.dumps(report["summary"], indent=2), flush=True)
    if not report["minimum_generated_target_met"]:
        raise SystemExit(
            "Generated-answer target was not met; provider fallbacks are not LLM successes."
        )


if __name__ == "__main__":
    main()
