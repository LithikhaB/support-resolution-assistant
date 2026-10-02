import json
from dataclasses import asdict

import pytest

from app.database.index_repository import vector_literal
from app.ingestion.artifacts import file_sha256
from app.ingestion.schema import SupportDocument
from app.retrieval.chunking import Chunk
from app.retrieval.corpus import CorpusDocument, PreparedCorpus


def item(doc_id="one"):
    doc = SupportDocument(
        doc_id=doc_id,
        doc_type="historical_response",
        title="Outage",
        body="The broadband connection keeps dropping.",
        response="Please confirm the model.",
    )
    content = doc.title + "\n\n" + doc.body
    return CorpusDocument(doc, [Chunk(doc_id, 0, content, 10, 0, len(content))])


def corpus_files(path, items):
    documents = path / "documents.jsonl"
    chunks = path / "chunks.jsonl"
    documents.write_text(
        "".join(x.document.model_dump_json() + "\n" for x in items), encoding="utf-8"
    )
    chunks.write_text(
        "".join(json.dumps(asdict(c)) + "\n" for x in items for c in x.chunks), encoding="utf-8"
    )
    manifest = {
        "source_sha256": file_sha256(documents),
        "output_sha256": file_sha256(chunks),
        "documents": len(items),
        "chunks": sum(len(x.chunks) for x in items),
        "max_tokens_including_special": 256,
    }
    (path / "chunks.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_corpus_streams_parent_evidence(tmp_path):
    corpus_files(tmp_path, [item("a"), item("b")])
    corpus = PreparedCorpus(tmp_path)
    assert corpus.validate() == {"a", "b"}
    records = list(corpus)
    assert records[0].document.response == "Please confirm the model."
    assert records[0].document.outcome_status == "unknown"


def test_hash_mismatch_rejected(tmp_path):
    corpus_files(tmp_path, [item()])
    (tmp_path / "chunks.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        PreparedCorpus(tmp_path)


@pytest.mark.parametrize(
    "failure", ["unknown_parent", "wrong_offset", "wrong_index", "too_long", "duplicates", "empty"]
)
def test_invalid_corpus_rejected_before_indexing(tmp_path, failure):
    record = item()
    if failure == "unknown_parent":
        record.chunks = [Chunk("missing", 0, "text", 2, 0, 4)]
    if failure == "wrong_offset":
        record.chunks = [Chunk("one", 0, "text", 2, 0, 4)]
    if failure == "wrong_index":
        record.chunks = [Chunk("one", 2, record.chunks[0].content, 10, 0, 46)]
    if failure == "too_long":
        c = record.chunks[0]
        record.chunks = [Chunk(c.doc_id, c.chunk_index, c.content, 300, c.char_start, c.char_end)]
    items = [] if failure == "empty" else [record, record] if failure == "duplicates" else [record]
    corpus_files(tmp_path, items)
    with pytest.raises(ValueError):
        PreparedCorpus(tmp_path).validate()


def test_fingerprint_tracks_evidence_changes_and_configuration():
    record = item()
    first = record.fingerprint("profile-a")
    assert first != record.fingerprint("profile-b")
    record.document.response = "New historical response"
    assert first != record.fingerprint("profile-a")


@pytest.mark.parametrize(
    "vector", [[1.0] * 383, [0.0] * 384, [float("nan")] * 384, [float("inf")] * 384, [1.0] * 384]
)
def test_invalid_vectors_rejected(vector):
    with pytest.raises(ValueError):
        vector_literal(vector)


def test_vector_serialization():
    values = [1.0] + [0.0] * 383
    assert json.loads(vector_literal(values)) == values
