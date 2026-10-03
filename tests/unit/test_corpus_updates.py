"""Ensure evolving data cannot silently delete history or contaminate evaluation splits."""

import json

import pytest

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


@pytest.mark.parametrize(
    "records",
    [
        [record("new"), record("new")],
        [record("new", metadata={"split": "test"})],
        [record("new", metadata={"kb_refs": ["missing"]})],
    ],
)
def test_invalid_update_leaves_no_staging_directory(tmp_path, records):
    current, incoming, output = tmp_path / "current", tmp_path / "incoming", tmp_path / "staged"
    write(current, [record("old")])
    write(incoming, records)
    with pytest.raises(ValueError):
        stage_update(current, incoming, output)
    assert not output.exists()
