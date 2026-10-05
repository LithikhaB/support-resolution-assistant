"""Token-bounded chunks preserving original text and document evidence links."""

from dataclasses import dataclass

from tokenizers import Tokenizer

from app.ingestion.schema import SupportDocument


@dataclass(frozen=True)
class Chunk:
    """Store a chunk identity, searchable text and exact source offsets."""

    doc_id: str
    chunk_index: int
    content: str
    token_count: int
    char_start: int
    char_end: int


class DocumentChunker:
    """Create token-bounded chunks with overlap and original character offsets."""

    def __init__(self, tokenizer: Tokenizer, max_tokens: int = 256, overlap_tokens: int = 32):

        self.tokenizer = Tokenizer.from_str(tokenizer.to_str())
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        special = self.tokenizer.num_special_tokens_to_add(False)
        if not 1 <= max_tokens <= 256 or max_tokens <= special:
            raise ValueError(
                "max_tokens must fit the 256-token model limit, including special tokens"
            )
        self.budget = max_tokens - special
        if not 0 <= overlap_tokens < self.budget:
            raise ValueError("overlap_tokens must be smaller than the content token budget")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    @staticmethod
    def retrieval_text(document: SupportDocument) -> str:
        """Use complaint or KB text for retrieval without leaking ticket resolutions."""
        body = document.body
        if document.metadata.get("procedure_version") in {2, 3}:
            body = document.metadata.get("retrieval_body", body)
        return f"{document.title.strip()}\n\n{body.strip()}"

    def chunk_document(self, document: SupportDocument) -> list[Chunk]:
        """Chunk searchable text while keeping response evidence on its parent document."""
        return self.chunk_text(document.doc_id, self.retrieval_text(document))

    def chunk_text(self, doc_id: str, text: str) -> list[Chunk]:
        """Split text at tokenizer boundaries and preserve verified character offsets."""
        if not doc_id.strip():
            raise ValueError("doc_id must be non-empty")
        if not text.strip():
            return []
        encoding = self.tokenizer.encode(text, add_special_tokens=False)
        offsets = encoding.offsets
        if not offsets:
            return []
        chunks: list[Chunk] = []
        start = 0
        while start < len(offsets):
            end = min(start + self.budget, len(offsets))

            while end > start:
                char_start = offsets[start][0]
                char_end = offsets[end - 1][1]
                content = text[char_start:char_end]
                count = len(self.tokenizer.encode(content).ids)
                if content.strip() and count <= self.max_tokens:
                    break
                end -= 1
            if end == start:
                raise ValueError("A token span cannot fit the configured chunk budget")
            chunks.append(Chunk(doc_id, len(chunks), content, count, char_start, char_end))
            if end == len(offsets):
                break

            start = max(start + 1, end - self.overlap_tokens)
        return chunks
