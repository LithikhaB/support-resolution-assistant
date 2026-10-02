import csv
import json

import pytest
from pydantic import ValidationError

from app.ingestion.artifacts import atomic_write, file_sha256
from app.ingestion.loader import load_tickets
from app.ingestion.schema import SupportDocument
from app.config.settings import Settings
from scripts import prepare_data


def write_csv(path, rows):
    columns = ["subject", "body", "answer", "language", "type", "priority", "queue", "version", "tag_1"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def row(**changes):
    return dict(subject="Outage", body="My broadband is down again.",
                answer="Please provide the router model.", language="en", type="Incident",
                priority="high", queue="Technical Support", version="51", tag_1="Network") | changes


def test_reply_is_not_a_resolution(tmp_path):
    path = tmp_path / "tickets.csv"
    write_csv(path, [row()])
    docs, drops = load_tickets(path)
    d = docs[0]
    assert not drops
    assert d.doc_type.value == "historical_response"
    assert d.response and d.resolution is None and d.outcome_status == "unknown"
    assert d.ticket_type == "incident" and d.priority.value == "high"
    assert d.intent is None and d.severity is None
    assert d.product is None and d.sentiment is None


def test_variants_and_source_provenance_survive_deduplication(tmp_path):
    path = tmp_path / "tickets.csv"
    write_csv(path, [row(), row(), row(answer="An engineer repaired the cable."), row(priority="low")])
    docs, drops = load_tickets(path)
    assert len(docs) == 3
    assert drops == {"duplicate_record": 1}
    assert len({d.metadata["complaint_group_id"] for d in docs}) == 1
    assert sorted(len(d.metadata["provenance"]) for d in docs) == [1, 1, 2]
    duplicate = next(d for d in docs if len(d.metadata["provenance"]) == 2)
    assert [p["source_row"] for p in duplicate.metadata["provenance"]] == [2, 3]


def test_ids_survive_row_reordering(tmp_path):
    path = tmp_path / "tickets.csv"
    rows = [row(), row(answer="A different response")]
    write_csv(path, rows)
    first, _ = load_tickets(path)
    write_csv(path, rows[::-1])
    second, _ = load_tickets(path)
    assert [d.doc_id for d in first] == [d.doc_id for d in second]


def test_invalid_input_and_drop_accounting(tmp_path):
    path = tmp_path / "tickets.csv"
    write_csv(path, [row(language="de"), row(body="short"), row(answer=" <br> "), row()])
    docs, drops = load_tickets(path)
    assert len(docs) == 1
    assert drops == {"non_english": 1, "body_too_short": 1, "missing_response": 1}
    path.write_text("body,answer\nhello,world\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing CSV columns"):
        load_tickets(path)


@pytest.mark.parametrize("extra", [
    {"doc_type": "resolved_ticket", "resolution": "Fixed"},
    {"doc_type": "resolved_ticket", "resolution": " ", "outcome_status": "verified_resolved"},
    {"doc_type": "resolved_ticket", "resolution": "Fixed", "outcome_status": "verified_resolved"},
    {"doc_type": "resolved_ticket", "resolution": "Fixed", "outcome_status": "verified_resolved", "metadata": {"outcome_evidence": " "}},
    {"doc_type": "historical_response", "response": " "},
    {"doc_type": "historical_response", "response": "Waiting", "resolution": ""},
    {"doc_type": "historical_response", "response": "Waiting", "resolution": "Fixed"},
    {"doc_type": "knowledge_base", "outcome_status": "verified_resolved"},
])
def test_invalid_evidence_is_rejected(extra):
    with pytest.raises(ValidationError):
        SupportDocument(doc_id="test", title="Test", body="A sufficiently long complaint", **extra)


def test_verified_evidence_and_kb_are_supported():
    common = dict(doc_id="test", title="Test", body="A sufficiently long complaint")
    d = SupportDocument(**common, doc_type="resolved_ticket", resolution="Cable replaced",
                        outcome_status="verified_resolved", metadata={"outcome_evidence": "Closure note 123"})
    assert d.resolution == "Cable replaced"
    assert SupportDocument(**common, doc_type="knowledge_base").resolution is None


def test_atomic_write_preserves_existing_file_on_failure(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text("original", encoding="utf-8")
    def broken():
        yield "partial"
        raise RuntimeError("interrupted")
    with pytest.raises(RuntimeError):
        atomic_write(path, broken())
    assert path.read_text() == "original"
    assert list(tmp_path.iterdir()) == [path]


def test_prepare_is_reproducible_and_checks_manifest(tmp_path, monkeypatch):
    settings = Settings(_env_file=None, data_dir=tmp_path)
    settings.raw_dir.mkdir()
    path = settings.raw_dir / "customer_support_tickets.csv"
    write_csv(path, [row(), row()])
    monkeypatch.setattr(prepare_data, "get_settings", lambda: settings)
    prepare_data.main()
    output = settings.processed_dir / "documents.jsonl"
    checksum = file_sha256(output)
    prepare_data.main()
    assert file_sha256(output) == checksum
    report = json.loads((settings.processed_dir / "manifest.json").read_text())
    assert report["documents"] == 1 and report["input_rows"] == 2
    assert report["source_revision"] is None
    assert report["source_provenance_status"] == "legacy_local_snapshot"
    path.with_suffix(".manifest.json").write_text(json.dumps({"sha256": "bad"}))
    with pytest.raises(ValueError, match="checksum"):
        prepare_data.main()
    assert file_sha256(output) == checksum
