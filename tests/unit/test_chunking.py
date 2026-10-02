import json

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from app.config.settings import Settings
from app.ingestion.schema import SupportDocument
from app.retrieval.chunking import DocumentChunker
from scripts import chunk_documents


@pytest.fixture
def tokenizer():

    tok = Tokenizer(models.WordLevel({"[UNK]": 0, "[CLS]": 1, "[SEP]": 2}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    return tok


def document(**extra):
    return SupportDocument(
        doc_id="example",
        doc_type="historical_response",
        title="Connection issue",
        body="My broadband disconnects every evening.",
        response="Please confirm your model.",
        **extra,
    )


def test_short_document_preserves_text_and_parent_evidence(tokenizer):
    doc = document()
    chunker = DocumentChunker(tokenizer)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].content == doc.title + "\n\n" + doc.body
    assert chunks[0].doc_id == doc.doc_id and chunks[0].chunk_index == 0
    assert doc.response not in chunks[0].content
    assert doc.response == "Please confirm your model."
    assert doc.outcome_status == "unknown"


def test_long_text_has_overlap_order_and_complete_coverage(tokenizer):
    text = " ".join(f"word{i}" for i in range(31))
    chunker = DocumentChunker(tokenizer, max_tokens=10, overlap_tokens=2)
    chunks = chunker.chunk_text("long", text)
    assert len(chunks) == 5
    assert [c.chunk_index for c in chunks] == list(range(5))
    assert all(c.token_count <= 10 for c in chunks)
    for chunk in chunks:
        assert text[chunk.char_start : chunk.char_end] == chunk.content
        assert chunk.token_count == len(tokenizer.encode(chunk.content).ids)
    for a, b in zip(chunks, chunks[1:]):
        assert a.content.split()[-2:] == b.content.split()[:2]
        assert a.char_start < b.char_start <= a.char_end
    assert chunks[0].char_start == 0 and chunks[-1].char_end == len(text)


def test_zero_overlap_and_exact_boundary(tokenizer):
    chunks = DocumentChunker(tokenizer, 10, 0).chunk_text("doc", " ".join(["word"] * 16))
    assert len(chunks) == 2
    assert all(c.token_count == 10 for c in chunks)


@pytest.mark.parametrize("text", ["", "  ", "\n\t"])
def test_empty_text(tokenizer, text):
    assert DocumentChunker(tokenizer).chunk_text("doc", text) == []


@pytest.mark.parametrize("limit,overlap", [(257, 1), (2, 0), (10, 8), (10, -1)])
def test_invalid_configuration(tokenizer, limit, overlap):
    with pytest.raises(ValueError):
        DocumentChunker(tokenizer, limit, overlap)


def test_empty_id(tokenizer):
    with pytest.raises(ValueError, match="doc_id"):
        DocumentChunker(tokenizer).chunk_text(" ", "Some text")


def test_unicode_paragraphs_are_preserved(tokenizer):
    text = "Café 网络\n\nRouter E102: disconnected!\nSecond paragraph."
    chunks = DocumentChunker(tokenizer).chunk_text("doc", text)
    assert chunks[0].content == text


def test_truncation_and_padding_do_not_hide_long_input(tokenizer):
    tokenizer.enable_truncation(max_length=8)
    tokenizer.enable_padding(length=32)
    chunks = DocumentChunker(tokenizer, 10, 2).chunk_text("doc", " ".join(["word"] * 30))
    assert len(chunks) > 1
    assert chunks[-1].char_end == len(" ".join(["word"] * 30))
    assert tokenizer.truncation is not None


def test_verified_resolution_is_kept_on_parent(tokenizer):
    doc = SupportDocument(
        doc_id="v",
        doc_type="resolved_ticket",
        title="Line fault",
        body="Broadband disconnects every evening.",
        resolution="Cable replaced",
        outcome_status="verified_resolved",
        metadata={"outcome_evidence": "closure note"},
    )
    chunks = DocumentChunker(tokenizer).chunk_document(doc)
    assert "Cable replaced" not in chunks[0].content
    assert chunks[0].doc_id == "v" and doc.resolution == "Cable replaced"


def test_cli_writes_reproducible_chunks_and_preserves_old_output_on_error(
    tmp_path, monkeypatch, tokenizer
):
    settings = Settings(_env_file=None, data_dir=tmp_path, corpus_dir=tmp_path)
    settings.processed_dir.mkdir()
    source = settings.processed_dir / "documents.jsonl"
    source.write_text(document().model_dump_json() + "\n", encoding="utf-8")
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    monkeypatch.setattr(chunk_documents, "get_settings", lambda: settings)
    monkeypatch.setattr(
        chunk_documents, "load_tokenizer", lambda *args: (tokenizer, tokenizer_path)
    )
    monkeypatch.setattr("sys.argv", ["chunk_documents"])
    chunk_documents.main()
    out = settings.processed_dir / "chunks.jsonl"
    original = out.read_bytes()
    chunk_documents.main()
    assert out.read_bytes() == original
    report = json.loads(out.with_suffix(".manifest.json").read_text())
    assert report["documents"] == 1 and report["chunks"] == 1
    source.write_text("invalid json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        chunk_documents.main()
    assert out.read_bytes() == original
