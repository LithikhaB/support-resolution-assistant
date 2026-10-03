"""Stage additive corpus updates without changing the active index or frozen evaluation data."""

from pathlib import Path

from app.ingestion.artifacts import atomic_write, file_sha256, write_json
from app.ingestion.schema import SupportDocument


def read_documents(path):
    """Validate every record and reject ambiguous repeated document identifiers."""
    records = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            document = SupportDocument.model_validate_json(line)
            if document.doc_id in records:
                raise ValueError("duplicate document identifier")
            records[document.doc_id] = document
    if not records:
        raise ValueError("document input is empty")
    return records


def stage_update(current, incoming, output):
    """Preserve all existing IDs and publish a new preparation manifest after validation."""
    current, incoming, output = Path(current), Path(incoming), Path(output)
    if output.exists():
        raise ValueError("choose a new staging directory")
    records, changes = read_documents(current), read_documents(incoming)
    if any(d.metadata.get("split") in {"dev", "test"} for d in changes.values()):
        raise ValueError("evaluation records cannot be imported into the retrieval corpus")
    updated = len(records.keys() & changes.keys())
    records.update(changes)
    known = set(records)
    for record in records.values():
        refs = record.metadata.get("kb_refs", [])
        if not isinstance(refs, list) or any(
            not isinstance(ref, str) or ref not in known for ref in refs
        ):
            raise ValueError("unresolved knowledge-base reference")
    output.mkdir(parents=True, exist_ok=False)
    documents = output / "documents.jsonl"
    atomic_write(documents, (records[key].model_dump_json() + "\n" for key in sorted(records)))
    manifest = {
        "version": "additive_update_v1",
        "documents": len(records),
        "added": len(changes) - updated,
        "updated": updated,
        "parent_sha256": file_sha256(current),
        "incoming_sha256": file_sha256(incoming),
        "output_sha256": file_sha256(documents),
        "categories": sorted({d.intent for d in records.values() if d.intent}),
        "activation": "Run chunk_documents for this directory, then index_documents with CORPUS_DIR set to its parent. Retrain and evaluate the classifier separately for new categories.",
    }
    write_json(output / "manifest.json", manifest)
    return manifest
