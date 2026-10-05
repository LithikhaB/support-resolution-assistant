import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from app.retrieval.chunking import DocumentChunker


@pytest.fixture
def tokenizer():

    tok = Tokenizer(models.WordLevel({"[UNK]": 0, "[CLS]": 1, "[SEP]": 2}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    return tok


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
