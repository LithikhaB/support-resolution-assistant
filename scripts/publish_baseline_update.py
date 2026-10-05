"""Publish v3.1 baseline investigations without modifying frozen v3 or held-out files."""

import argparse
import json
import shutil
from pathlib import Path

from app.ingestion.artifacts import atomic_write, file_sha256, write_json
from scripts.publish_telecom_v3 import article

BASELINES = {
    "GENERAL_OUTAGE": {
        "category": "broadband_outage",
        "template": "syn_kb_BB01",
        "title": "Broadband unavailable: establish the fault before choosing a repair",
        "customer_checks": [
            "Ask whether all devices lack internet or only one, and whether a wired device is also affected; record when service stopped.",
            "Record the router and fibre-box light labels and check whether nearby customers are affected; do not repeat completed restarts or disconnect optical cabling.",
        ],
        "agent_checks": "Check provider incident monitoring and account status, then ask authorised support to test the line and subscriber session using the recorded symptoms. Record the actual fault finding before selecting a repair.",
        "diagnostic_gate": "authorised diagnostics establish a specific line or subscriber-session fault and identify the corresponding approved repair procedure",
        "action": "route the recorded findings to the authorised team for that confirmed fault and follow its approved procedure; do not promise a repair time",
        "completion": "the affected wired and wireless devices can browse again and no repeat loss is reported during the agreed observation period",
    },
    "GENERAL_SLOW": {
        "category": "slow_broadband",
        "template": "syn_kb_SP04",
        "title": "Persistent broadband slowness: establish a controlled baseline",
        "customer_checks": [
            "Ask whether one device or all devices are slow. Compare a wired speed test with Wi-Fi if available, and record the subscribed speed and test time.",
            "Ask about active downloads, uploads, cameras or backups. With consent, briefly pause an identified nonessential transfer and repeat the same test; do not disable security services.",
        ],
        "agent_checks": "If controlled tests remain slow, check provider incident monitoring, line/session status and the subscribed speed profile. Record whether a transfer, wireless-only problem or provider fault explains the measured result; do not assume evening congestion.",
        "diagnostic_gate": "the controlled test remains slow after nonessential transfers are paused and authorised provider diagnostics establish a specific line or provisioning fault",
        "action": "have the authorised provider team apply the approved correction for the confirmed fault and repeat the same controlled test; do not book a physical repair without line-fault evidence",
        "completion": "repeat measurements are consistent with the subscribed service under the test conditions and the customer's affected applications work normally",
    },
}


def publish(source, output):
    if output.exists():
        raise ValueError("Corpus versions are immutable; choose a new output directory")
    rows = [
        json.loads(s)
        for s in (source / "processed/documents.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    by_id = {r["doc_id"]: r for r in rows}
    for name, authored in BASELINES.items():
        row = json.loads(json.dumps(by_id[authored["template"]]))
        row.update(doc_id=f"syn_kb_{name}", intent=authored["category"], product="broadband")
        row["metadata"].update(
            category=authored["category"],
            scenario_family=f"baseline_{name}",
            baseline_for=authored["category"],
            kb_refs=[],
        )
        row["metadata"]["procedure"]["restriction"] = [
            "Do not repeat completed checks, infer a cause from similarity, handle optical cabling or promise an unverified repair."
        ]
        row["metadata"]["retrieval_body"] = (
            "Broadband internet is unavailable on devices."
            if name == "GENERAL_OUTAGE"
            else "Broadband internet has been very slow since morning. Persistent daytime slowness needs controlled wired and wireless baseline measurements."
        )
        rows.append(article(row, authored))
    rows.sort(key=lambda r: r["doc_id"])
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
    for row in rows:
        row["metadata"]["dataset_version"] = "telecom-synthetic-3.1.0"
    for name, records in (
        ("processed/documents.jsonl", rows),
        ("knowledge_base.jsonl", [r for r in rows if r["doc_type"] == "knowledge_base"]),
        ("tickets.jsonl", [r for r in rows if r["doc_type"] == "resolved_ticket"]),
    ):
        atomic_write(
            output / name,
            (json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records),
        )
    manifest = {
        "dataset_version": "telecom-synthetic-3.1.0",
        "documents": len(rows),
        "source_sha256": file_sha256(source / "processed/documents.jsonl"),
        "output_sha256": file_sha256(output / "processed/documents.jsonl"),
        "expert_reviewed": False,
        "source": "AI-authored synthetic baseline investigations added to unchanged v3 source procedures",
    }
    write_json(output / "processed/manifest.json", manifest)
    write_json(
        output / "quality_report.json",
        {
            **manifest,
            "knowledge_base_articles": 62,
            "simulated_training_tickets": 240,
            "heldout_tickets_in_index": 0,
            "baseline_authoring": BASELINES,
            "file_sha256": {p.name: file_sha256(p) for p in output.glob("*.jsonl")},
        },
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/synthetic/telecom_v3"))
    parser.add_argument("--output", type=Path, default=Path("data/synthetic/telecom_v3_1"))
    args = parser.parse_args()
    print(json.dumps(publish(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
