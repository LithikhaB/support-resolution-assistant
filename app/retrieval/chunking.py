"""Token-bounded chunks preserving original text and document evidence links."""
from dataclasses import dataclass

from tokenizers import Tokenizer

from app.ingestion.schema import SupportDocument


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_index: int
    content: str
    token_count: int  # Includes the model's special tokens.
    char_start: int
    char_end: int


class DocumentChunker:
    def __init__(self, tokenizer: Tokenizer, max_tokens: int = 256, overlap_tokens: int = 32):
        # Copy before disabling truncation: do not mutate the caller's tokenizer.
        self.tokenizer = Tokenizer.from_str(tokenizer.to_str())
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        special = self.tokenizer.num_special_tokens_to_add(False)
        if not 1 <= max_tokens <= 256 or max_tokens <= special:
            raise ValueError("max_tokens must fit the 256-token model limit, including special tokens")
        self.budget = max_tokens - special
        if not 0 <= overlap_tokens < self.budget:
            raise ValueError("overlap_tokens must be smaller than the content token budget")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    @staticmethod
    def retrieval_text(document: SupportDocument) -> str:
        # Replies/resolutions remain on the parent document, never misrepresented
        # as complaint text. KB documents use the same title/body convention.
        return f"{document.title.strip()}\n\n{document.body.strip()}"

    def chunk_document(self, document: SupportDocument) -> list[Chunk]:
        return self.chunk_text(document.doc_id, self.retrieval_text(document))

    def chunk_text(self, doc_id: str, text: str) -> list[Chunk]:
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
            # Re-encoding a substring can change WordPiece boundaries. Verify the
            # actual emitted string fits, instead of trusting the original window.
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
            # Usually exactly overlap_tokens; reduced only to guarantee progress
            # when a boundary adjustment produced an unusually small chunk.
            start = max(start + 1, end - self.overlap_tokens)
        return chunks
