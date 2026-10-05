"""Publish immutable v3 procedures and linked simulated tickets; never read held-out queries."""

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

from app.ingestion.artifacts import atomic_write, file_sha256, write_json
from app.ingestion.schema import SupportDocument


def article(row, authored):
    row = json.loads(json.dumps(row))
    previous = row["metadata"]["procedure"]
    checks = authored["customer_checks"]
    if len(checks) != 2 or any(not isinstance(s, str) or len(s) < 20 for s in checks):
        raise ValueError("Each article needs two issue-specific checks")

    def clause(text):
        text = text.strip().rstrip(".")
        # Keep acronyms intact while joining sentence fragments into instructions.
        return text[0].lower() + text[1:] if not text.split()[0].isupper() else text

    gate = clause(authored.get("diagnostic_gate", previous["diagnostic_gate"]))
    action = clause(authored.get("action", previous["fix_if_confirmed"][0]))
    completion = clause(authored.get("completion", previous["completion"][0]))
    agent = authored.get(
        "agent_checks",
        "Ask provider support to verify these required findings before choosing a repair: " + gate,
    )
    fix = f"Only if support confirms that {gate}: {action}."
    finish = f"After the authorised remedy, verify that {completion}."
    steps = [
        {"id": 1, "phase": "customer_check", "text": checks[0]},
        {"id": 2, "phase": "customer_check", "text": checks[1]},
        {"id": 3, "phase": "agent_check", "text": agent},
        {"id": 4, "phase": "conditional_fix", "text": fix},
        {"id": 5, "phase": "completion", "text": finish},
    ]
    for step in steps[:2]:
        if re.search(r"\bwired (?:device|test)|\bEthernet\b", step["text"], re.I):
            step["skip_if_fact"] = "wired_connection"
        if "pending" in step["text"] and "settled" in step["text"]:
            step["skip_if_fact"] = "billing_status"
    escalation = "If the required findings are not established or service still fails after the authorised remedy, refer the recorded results to specialist support; do not guess a repair."
    procedure = dict(
        previous,
        verify=[checks[0]],
        customer_checks=checks,
        agent_checks=[agent],
        diagnostic_gate=gate,
        fix_if_confirmed=[action],
        completion=[completion],
        escalate_if=[escalation],
        steps=steps,
    )
    row["title"] = authored.get("title", row["title"].replace(". — procedure", ""))
    row["body"] = "\n".join(
        [
            "AI-authored synthetic procedure; fictional provider; not telecom-expert reviewed.",
            f"Scope: {row['product']}; support category: {row['intent']}.",
            "Verify: " + checks[0],
            "Customer checks: " + " ".join(checks),
            "Agent checks: " + agent,
            "Diagnostic gate: " + gate,
            "Only if that finding is established: " + action,
            "Escalate if: " + escalation,
            "Restriction: " + " ".join(previous["restriction"]),
            "Completion: " + completion,
            *(f"Step {s['id']}: {s['text']}" for s in steps),
        ]
    )
    row["metadata"].update(
        dataset_version="telecom-synthetic-3.0.0",
        version=3,
        procedure_version=3,
        authorship="AI-authored synthetic; no telecom expert review",
        procedure=procedure,
    )
    # Search remains symptom-focused; checklists/resolutions are evidence, not query leakage.
    row["metadata"]["retrieval_body"] = row["metadata"]["retrieval_body"] + "\n" + row["title"]
    return SupportDocument.model_validate(row).model_dump(mode="json")


def publish(source, authoring, output):
    if output.exists():
        raise ValueError("Existing corpus versions are immutable; choose a new output directory")
    authored = json.loads(authoring.read_text(encoding="utf-8"))
    rows = [
        json.loads(s)
        for s in (source / "processed/documents.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    kb_rows = [r for r in rows if r["doc_type"] == "knowledge_base"]
    if set(authored["procedures"]) != {r["doc_id"].removeprefix("syn_kb_") for r in kb_rows}:
        raise ValueError("Every reference article must have authored checks")
    articles = {
        r["doc_id"]: article(r, authored["procedures"][r["doc_id"].removeprefix("syn_kb_")])
        for r in kb_rows
    }
    tickets = []
    for row in rows:
        if row["doc_type"] == "knowledge_base":
            continue
        if row["metadata"].get("split") != "train":
            raise ValueError("Only training tickets may enter the retrieval corpus")
        row["metadata"].update(dataset_version="telecom-synthetic-3.0.0")
        if row["doc_type"] == "resolved_ticket":
            if row["outcome_status"] != "simulated_resolved":
                raise ValueError("Cannot rewrite real resolved outcomes as authored simulations")
            ref = row["metadata"]["kb_refs"][0]
            procedure = articles[ref]["metadata"]["procedure"]
            row["metadata"]["ticket_id"] = row["doc_id"]
            row["metadata"]["resolution_version"] = 3
            row["resolution"] = "\n".join(
                [
                    f"Ticket {row['doc_id']}; AI-authored synthetic; outcome: simulated_resolved.",
                    "Simulated finding: " + procedure["diagnostic_gate"],
                    *(f"Step {s['id']}: {s['text']}" for s in procedure["steps"]),
                    "Simulated completion: " + procedure["completion"][0],
                ]
            )
            tickets.append(row)
    records = [articles[r["doc_id"]] if r["doc_type"] == "knowledge_base" else r for r in rows]
    for r in records:
        SupportDocument.model_validate(r)
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
    for name, data in (
        ("knowledge_base.jsonl", list(articles.values())),
        ("tickets.jsonl", tickets),
        ("processed/documents.jsonl", records),
    ):
        atomic_write(
            output / name, (json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in data)
        )
    manifest = {
        "dataset_version": "telecom-synthetic-3.0.0",
        "documents": len(records),
        "source_sha256": file_sha256(source / "processed/documents.jsonl"),
        "output_sha256": file_sha256(output / "processed/documents.jsonl"),
        "authoring_sha256": file_sha256(authoring),
        "expert_reviewed": False,
        "source": "AI-authored synthetic procedures and matching simulated training-ticket resolutions",
    }
    write_json(output / "processed/manifest.json", manifest)
    quality = {
        **manifest,
        "document_types": dict(Counter(r["doc_type"] for r in records)),
        "heldout_tickets_in_index": 0,
        "reference_urls": authored["reference_urls"],
        "file_sha256": {p.name: file_sha256(p) for p in output.glob("*.jsonl")},
    }
    write_json(output / "quality_report.json", quality)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/synthetic/telecom_v2"))
    parser.add_argument(
        "--authoring",
        type=Path,
        default=Path("data/synthetic/telecom_v3_authoring/procedures.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/synthetic/telecom_v3"))
    args = parser.parse_args()
    print(json.dumps(publish(args.source, args.authoring, args.output), indent=2))


if __name__ == "__main__":
    main()
