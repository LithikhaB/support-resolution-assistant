"""Publish AI-authored procedure v2 from reference KB only; never read evaluation queries."""

import argparse
import json
import re
import shutil
from pathlib import Path

from app.ingestion.artifacts import atomic_write, file_sha256, write_json


def enrich(row):
    """Retain the diagnostic gate and conditional remedy without inventing findings."""
    row = json.loads(json.dumps(row))
    body = row["body"]
    gate = re.search(r"^Diagnostic gate: (.+)$", body, re.M).group(1)
    action = re.search(r"^Only if that finding is established: (.+)$", body, re.M).group(1)
    restriction = re.search(r"^Restriction: (.+)$", body, re.M).group(1)
    completion = re.search(r"^Completion criterion in this simulation: (.+)$", body, re.M).group(1)
    product = row["product"]
    verify = "Record the affected service, onset, frequency and customer impact. Review completed checks and their results."
    customer = "Ask the customer for the displayed symptom or error; do not request passwords, OTPs or payment credentials."
    if product in {"fibre_broadband", "home_wifi", "router"}:
        customer = "Ask whether a wired device is affected during the same failure. Record status lights without treating their colour as a verified line test."
    elif product == "billing":
        customer = "Ask which invoice or charge is affected and whether each payment is pending or settled; collect only a non-sensitive description."
    elif product.startswith("mobile") or product in {"sim", "esim"}:
        customer = "Ask which of calls, texts and mobile data are affected, where this occurs and what error appears."
    elif product == "iptv":
        customer = "Ask which channels are affected and whether the screen shows an error, buffering or no picture."
    agent = (
        "Use authorized provider records or diagnostics to check every part of this gate: " + gate
    )
    escalate = "If the gate is unconfirmed, contradictory or outside the agent's authority, route for specialist investigation with the observations; do not apply the remedy."
    row["procedure"] = {
        "verify": [verify],
        "customer_checks": [customer],
        "agent_checks": [agent],
        "diagnostic_gate": gate,
        "fix_if_confirmed": [action],
        "escalate_if": [escalate],
        "restriction": [restriction],
        "completion": [completion],
    }
    row["body"] = "\n".join(
        [
            "AI-authored synthetic procedure; fictional provider; not expert reviewed.",
            f"Scope: {product}; support category: {row['intent']}.",
            "Verify: " + verify,
            "Customer checks: " + customer,
            "Agent checks: " + agent,
            "Diagnostic gate: " + gate,
            "Only if that finding is established: " + action,
            "Escalate if: " + escalate,
            "Restriction: " + restriction,
            "Completion: " + completion,
        ]
    )
    row["metadata"].update(
        dataset_version="telecom-synthetic-2.0.0",
        procedure_version=2,
        version=2,
        authorship="AI-authored synthetic; no telecom expert review",
    )
    row["metadata"]["procedure"] = row.pop("procedure")
    row["metadata"]["retrieval_body"] = body
    return row


def publish(source, output):
    """Create a new version, preserving splits and excluding held-out tickets from the index."""
    if output.exists():
        raise ValueError("Choose a new corpus directory; existing versions are immutable")
    documents = [
        json.loads(line)
        for line in (source / "processed/documents.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    rows = [enrich(row) if row["doc_type"] == "knowledge_base" else row for row in documents]
    if any(
        r["doc_type"] != "knowledge_base" and r["metadata"]["split"] in {"dev", "test"}
        for r in rows
    ):
        raise ValueError("Held-out tickets cannot enter the index")
    output.mkdir(parents=True)
    for name in (
        "train.jsonl",
        "dev.jsonl",
        "test.jsonl",
        "scenarios.psv",
        "training_paraphrases.json",
        "challenge_queries.jsonl",
    ):
        shutil.copyfile(source / name, output / name)
    for path, records in (
        (output / "knowledge_base.jsonl", [r for r in rows if r["doc_type"] == "knowledge_base"]),
        (output / "processed/documents.jsonl", rows),
    ):
        atomic_write(
            path, (json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records)
        )
    report = {
        "dataset_version": "telecom-synthetic-2.0.0",
        "documents": len(rows),
        "output_sha256": file_sha256(output / "processed/documents.jsonl"),
        "source_sha256": file_sha256(source / "processed/documents.jsonl"),
        "source": "AI-authored synthetic procedure enrichment; gates and remedies retained from v1 reference KB",
        "expert_reviewed": False,
    }
    write_json(output / "processed/manifest.json", report)
    quality = json.loads((source / "quality_report.json").read_text(encoding="utf-8"))
    quality.update(dataset_version=report["dataset_version"], procedure_authorship=report["source"])
    quality["file_sha256"].update(
        {
            "knowledge_base.jsonl": file_sha256(output / "knowledge_base.jsonl"),
            "processed/documents.jsonl": report["output_sha256"],
        }
    )
    write_json(output / "quality_report.json", quality)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/synthetic/telecom_v1"))
    parser.add_argument("--output", type=Path, default=Path("data/synthetic/telecom_v2"))
    args = parser.parse_args()
    print(json.dumps(publish(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
