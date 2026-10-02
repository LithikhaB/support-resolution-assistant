"""Validate synthetic evidence, relevance references and family-separated splits."""

from collections import Counter, defaultdict

from app.ingestion.schema import SupportDocument


def validate_dataset(artifacts: dict[str, list[dict]]) -> dict:
    """Reject provenance, reference, split and severity inconsistencies before writing."""
    docs = [SupportDocument.model_validate(row) for row in artifacts["documents"]]
    if len({doc.doc_id for doc in docs}) != len(docs):
        raise ValueError("Duplicate corpus document IDs")
    families = {
        split: {r["scenario_family"] for r in artifacts[split]}
        for split in ("train", "dev", "test")
    }
    if any(
        families[a] & families[b] for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))
    ):
        raise ValueError("Scenario-family leakage across splits")
    held_out = families["dev"] | families["test"]
    if any(
        d.doc_type.value != "knowledge_base" and d.metadata["scenario_family"] in held_out
        for d in docs
    ):
        raise ValueError("Held-out ticket present in retrieval corpus")
    kb_ids = {r["doc_id"] for r in artifacts["knowledge_base"]}
    queries = []
    for split in ("train", "dev", "test"):
        for row in artifacts[split]:
            if row["split"] != split:
                raise ValueError("Query split disagrees with its artifact")
            if not row["relevant_kb_ids"] or not set(row["relevant_kb_ids"]) <= kb_ids:
                raise ValueError("Unresolvable KB relevance reference")
            if not set(row["contrast_candidate_ids"]) <= kb_ids or set(
                row["contrast_candidate_ids"]
            ) & set(row["relevant_kb_ids"]):
                raise ValueError("Invalid contrast references")
            queries.append(row["query"])
    if len(set(queries)) != len(queries):
        raise ValueError("Exact duplicate queries")

    severity_by_family = defaultdict(set)
    for row in artifacts["tickets"]:
        document = SupportDocument.model_validate(row)
        if any(ref not in kb_ids for ref in document.metadata.get("kb_refs", [])):
            raise ValueError("Ticket references missing KB")
        if document.outcome_status == "simulated_resolved":
            severity_by_family[document.metadata["scenario_family"]].add(document.severity.value)
    if any(len(values) != 1 for values in severity_by_family.values()):
        raise ValueError("Severity depends on tone within a scenario family")
    return {
        "checks_passed": [
            "schema",
            "unique_ids",
            "split_family_disjointness",
            "no_heldout_ticket_in_corpus",
            "kb_references",
            "unique_queries",
            "severity_independent_of_tone",
        ],
        "counts": {key: len(value) for key, value in artifacts.items()},
        "families_per_split": {k: len(v) for k, v in families.items()},
        "categories": dict(Counter(r["labels"]["intent"] for r in artifacts["train"])),
        "limitations": [
            "AI-authored scenarios; no telecom expert review or real-world outcome verification.",
            "Eight controlled variants per family are not independent incidents.",
            "Reference KB covers held-out families; this is unseen-ticket evaluation, not unseen-knowledge evaluation.",
            "Test data is visible for audit; freeze it before tuning. Independent human-authored tests remain necessary.",
        ],
    }
