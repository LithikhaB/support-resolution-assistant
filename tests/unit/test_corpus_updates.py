"""Ensure evolving data cannot silently delete history or contaminate evaluation splits."""

import json

from app.ingestion.updates import stage_update


def write(path, records):
    """Write minimal valid KB fixtures."""
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def record(identifier, **updates):
    """Create a schema-valid article with an extensible category."""
    return {
        "doc_id": identifier,
        "doc_type": "knowledge_base",
        "title": "Guide",
        "body": "A sufficiently long support procedure.",
        "intent": "new_service",
        **updates,
    }


def test_staging_preserves_existing_ids_and_active_file(tmp_path):
    current, incoming = tmp_path / "current.jsonl", tmp_path / "incoming.jsonl"
    write(current, [record("old")])
    write(incoming, [record("new")])
    before = current.read_bytes()
    result = stage_update(current, incoming, tmp_path / "staged")
    assert result["added"] == 1 and result["documents"] == 2
    assert current.read_bytes() == before
    assert result["categories"] == ["new_service"]
    from scripts.serve import install_missing_corpus

    bundled_root = tmp_path / "bundled"
    bundled = bundled_root / "telecom_v3_1"
    bundled.mkdir(parents=True)
    (bundled / "knowledge_base.jsonl").write_text("new version")
    target = tmp_path / "volume" / "telecom_v3_1"
    install_missing_corpus(target, bundled_root)
    assert (target / "knowledge_base.jsonl").read_text() == "new version"
    (bundled / "knowledge_base.jsonl").write_text("must not replace frozen data")
    install_missing_corpus(target, bundled_root)
    assert (target / "knowledge_base.jsonl").read_text() == "new version"
