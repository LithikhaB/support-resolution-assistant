import pytest
from pydantic import ValidationError

from app.ingestion.artifacts import atomic_write
from app.ingestion.schema import SupportDocument


@pytest.mark.parametrize(
    "extra",
    [
        {"doc_type": "resolved_ticket", "resolution": "Fixed"},
        {"doc_type": "resolved_ticket", "resolution": " ", "outcome_status": "verified_resolved"},
        {
            "doc_type": "resolved_ticket",
            "resolution": "Fixed",
            "outcome_status": "verified_resolved",
        },
        {
            "doc_type": "resolved_ticket",
            "resolution": "Fixed",
            "outcome_status": "verified_resolved",
            "metadata": {"outcome_evidence": " "},
        },
        {"doc_type": "historical_response", "response": " "},
        {"doc_type": "historical_response", "response": "Waiting", "resolution": ""},
        {"doc_type": "historical_response", "response": "Waiting", "resolution": "Fixed"},
        {"doc_type": "knowledge_base", "outcome_status": "verified_resolved"},
    ],
)
def test_invalid_evidence_is_rejected(extra):
    with pytest.raises(ValidationError):
        SupportDocument(doc_id="test", title="Test", body="A sufficiently long complaint", **extra)


def test_verified_evidence_and_kb_are_supported():
    common = dict(doc_id="test", title="Test", body="A sufficiently long complaint")
    d = SupportDocument(
        **common,
        doc_type="resolved_ticket",
        resolution="Cable replaced",
        outcome_status="verified_resolved",
        metadata={"outcome_evidence": "Closure note 123"},
    )
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
