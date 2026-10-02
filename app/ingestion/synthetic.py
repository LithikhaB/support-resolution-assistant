"""Deterministic, authored scenario corpus; no model/API calls or random fixes."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from app.ingestion.artifacts import atomic_write, file_sha256, write_json
from app.ingestion.schema import SupportDocument
from app.ingestion.validation import validate_dataset as validate_dataset

VERSION = "telecom-synthetic-1.0.0"
SOURCE = "Fictional Northstar Telecom / authored synthetic scenarios"
CATALOG = Path(__file__).resolve().parents[2] / "data/synthetic/telecom_v1/scenarios.psv"
MOODS = {
    "neutral": "Please explain the next step.",
    "concerned": "I am worried about this and would appreciate an update.",
    "frustrated": "I am frustrated with this problem and want a clear explanation.",
    "angry": "I am angry about this experience and want it addressed.",
}
HELDOUT_MOODS = {
    "dev": {
        "neutral": "Could you check this when you have the details?",
        "concerned": "I am anxious that this will disrupt my plans.",
        "frustrated": "Getting this sorted has become really tiresome.",
        "angry": "This level of service is unacceptable to me.",
    },
    "test": {
        "neutral": "I am reporting the observations so you can investigate.",
        "concerned": "I am uneasy about what might happen if this continues.",
        "frustrated": "I am fed up with dealing with this issue.",
        "angry": "Your service has let me down badly; I am furious.",
    },
}


def load_scenarios(path: Path = CATALOG) -> list[dict]:
    """Validate and order the authored catalog so row order cannot change data splits."""
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="|"))
    if len(rows) != 60 or len({r["family_id"] for r in rows}) != 60:
        raise ValueError("Expected 60 unique authored scenario families")
    counts = Counter(r["category"] for r in rows)
    if len(counts) != 15 or set(counts.values()) != {4}:
        raise ValueError("Expected four families for each of 15 categories")
    for row in rows:
        if None in row or any(not isinstance(v, str) or not v.strip() for v in row.values()):
            raise ValueError("Malformed or empty scenario fields")
    return sorted(rows, key=lambda row: row["family_id"])


def build_dataset(scenarios: list[dict]) -> dict[str, list[dict]]:
    """Build causally matched tickets and family-separated query splits."""
    artifacts = {
        name: [] for name in ("tickets", "knowledge_base", "train", "dev", "test", "documents")
    }
    seen_categories = Counter()
    for sc in sorted(scenarios, key=lambda row: row["family_id"]):
        category = sc["category"]
        position = seen_categories[category]
        seen_categories[category] += 1
        split = ("train", "train", "dev", "test")[position]
        family = sc["family_id"]
        kb_id = f"syn_kb_{family}"
        common = dict(
            is_synthetic=True,
            source=SOURCE,
            dataset_version=VERSION,
            scenario_family=family,
            category=category,
            queue="technical_support",
            review_status="authored_and_automatically_checked_not_expert_reviewed",
        )
        if category in {"billing_dispute", "payment_restoration"}:
            common["queue"] = "billing_support"
        kb_body = "\n".join(
            [
                "Fictional provider procedure; not an operational instruction for a real network.",
                f"Scope: {sc['product']}; support category: {category}.",
                "First inspect the reported checks and do not repeat an already completed check without a reason.",
                "Diagnostic gate: " + sc["diagnostic"],
                "Only if that finding is established: " + sc["action"],
                "If the finding is absent, ask for missing observations or route for investigation; do not assume the cause.",
                "Completion criterion in this simulation: "
                + sc["verification"].removeprefix("Simulated follow-up: "),
                "Restriction: " + sc["avoid"],
            ]
        )
        kb = SupportDocument(
            doc_id=kb_id,
            doc_type="knowledge_base",
            title=sc["cause"] + " — procedure",
            body=kb_body,
            intent=category,
            product=sc["product"],
            metadata=common
            | {
                "version": 1,
                "valid_from": "2026-10-01",
                "authority": "fictional_provider_policy",
                "split": "reference",
            },
        )
        artifacts["knowledge_base"].append(kb.model_dump(mode="json"))
        artifacts["documents"].append(kb.model_dump(mode="json"))
        related = [
            f"syn_kb_{other['family_id']}"
            for other in sorted(scenarios, key=lambda row: row["family_id"])
            if other["category"] == category and other["family_id"] != family
        ]
        for wording, field in enumerate(("complaint", "paraphrase")):
            for mood_index, (mood, closing) in enumerate(HELDOUT_MOODS.get(split, MOODS).items()):
                variant = wording * len(MOODS) + mood_index + 1
                ticket_id = f"syn_ticket_{family}_{variant:02}"
                query = f"{sc[field]} Already checked: {sc['attempted']} {closing}"

                metadata = common | {
                    "split": split,
                    "kb_refs": [kb_id],
                    "label_basis": "authored_scenario",
                    "attempted_steps": [sc["attempted"]],
                    "diagnostic_findings": [sc["diagnostic"]],
                    "root_cause": sc["cause"],
                    "resolution_steps": [sc["action"]],
                    "outcome_evidence": sc["verification"],
                    "forbidden_action": sc["avoid"],
                    "severity_basis": "service impact in the scenario; independent of tone",
                    "complaint_group_id": hashlib.sha256(query.casefold().encode()).hexdigest(),
                }
                ticket = SupportDocument(
                    doc_id=ticket_id,
                    doc_type="resolved_ticket",
                    title=category.replace("_", " ").capitalize() + " support case",
                    body=query,
                    resolution=f"Finding: {sc['diagnostic']} Action: {sc['action']} {sc['verification']}",
                    outcome_status="simulated_resolved",
                    intent=category,
                    product=sc["product"],
                    severity=sc["severity"],
                    sentiment=mood,
                    ticket_type="request"
                    if category in {"billing_dispute", "number_porting"}
                    else "incident",
                    metadata=metadata,
                )
                artifacts["tickets"].append(ticket.model_dump(mode="json"))
                if split == "train":
                    artifacts["documents"].append(ticket.model_dump(mode="json"))
                artifacts[split].append(
                    dict(
                        query_id=f"query_{family}_{variant:02}",
                        query=query,
                        scenario_family=family,
                        split=split,
                        is_synthetic=True,
                        labels={
                            "intent": category,
                            "product": sc["product"],
                            "severity": sc["severity"],
                            "sentiment": mood,
                        },
                        attempted_steps=[sc["attempted"]],
                        expected_behavior="diagnose_before_action",
                        relevant_kb_ids=[kb_id],
                        contrast_candidate_ids=related,
                        relevance_note="Author-assigned KB relevance, not exhaustive human judgments. Contrast candidates need review before use as training negatives.",
                        required_diagnostic=sc["diagnostic"],
                        forbidden_action=sc["avoid"],
                    )
                )

        if split == "train":
            pending = SupportDocument(
                doc_id=f"syn_pending_{family}",
                doc_type="historical_response",
                title="Investigation pending",
                body=sc["paraphrase"] + " No further diagnostic result is available.",
                response="The agent requested the missing observations; no action or successful outcome is recorded.",
                intent=category,
                product=sc["product"],
                metadata=common
                | {"split": split, "kb_refs": [kb_id], "case_status": "awaiting_diagnostics"},
            )
            artifacts["tickets"].append(pending.model_dump(mode="json"))
            artifacts["documents"].append(pending.model_dump(mode="json"))
    artifacts["documents"].sort(key=lambda row: row["doc_id"])
    return artifacts


def write_dataset(output: Path, catalog: Path = CATALOG) -> dict:
    """Validate and reproducibly write the corpus, query splits and checksum manifests."""
    artifacts = build_dataset(load_scenarios(catalog))
    report = validate_dataset(artifacts)
    challenge_path = catalog.with_name("challenge_queries.jsonl")
    challenges = [
        json.loads(line)
        for line in challenge_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len({row["query_id"] for row in challenges}) != len(challenges) or not all(
        row.get("is_synthetic") is True
        and row.get("query")
        and row.get("expected_behavior")
        and row.get("forbidden_action")
        for row in challenges
    ):
        raise ValueError("Invalid challenge cases")
    artifacts["challenge_queries"] = challenges
    report["counts"]["challenge_queries"] = len(challenges)
    files = {}
    for name, records in artifacts.items():
        relative = (
            Path("processed/documents.jsonl") if name == "documents" else Path(name + ".jsonl")
        )
        path = output / relative
        atomic_write(
            path, (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records)
        )
        files[str(relative).replace("\\", "/")] = file_sha256(path)
    report.update(dataset_version=VERSION, catalog_sha256=file_sha256(catalog), file_sha256=files)
    write_json(output / "quality_report.json", report)
    write_json(
        output / "processed/manifest.json",
        {
            "dataset_version": VERSION,
            "documents": len(artifacts["documents"]),
            "output_sha256": files["processed/documents.jsonl"],
            "source_sha256": file_sha256(catalog),
            "source": SOURCE,
        },
    )
    return report
