"""Verify Phase B artifacts and stream one document with its chunks at a time."""
import json
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

from app.ingestion.artifacts import file_sha256
from app.ingestion.loader import digest
from app.ingestion.schema import SupportDocument
from app.retrieval.chunking import Chunk, DocumentChunker


@dataclass
class CorpusDocument:
    document: SupportDocument
    chunks: list[Chunk]

    def fingerprint(self, config_hash: str) -> str:
        return digest({"document": self.document.model_dump(mode="json"),
                       "chunks": [c.content for c in self.chunks], "config_hash": config_hash})


class PreparedCorpus:
    def __init__(self, directory: Path):
        self.documents_path = directory / "documents.jsonl"
        self.chunks_path = directory / "chunks.jsonl"
        self.manifest = json.loads((directory / "chunks.manifest.json").read_text(encoding="utf-8"))
        self.verify_hashes()

    def verify_hashes(self) -> None:
        if file_sha256(self.documents_path) != self.manifest["source_sha256"]:
            raise ValueError("Document checksum mismatch; rerun preparation/chunking")
        if file_sha256(self.chunks_path) != self.manifest["output_sha256"]:
            raise ValueError("Chunk checksum mismatch; rerun chunking")

    def __iter__(self):
        with self.documents_path.open(encoding="utf-8") as docs, self.chunks_path.open(encoding="utf-8") as chunks:
            groups = iter(groupby((Chunk(**json.loads(line)) for line in chunks), key=lambda c: c.doc_id))
            previous = ""
            for line in docs:
                doc = SupportDocument.model_validate_json(line)
                if doc.doc_id <= previous:
                    raise ValueError("Documents must have unique IDs sorted in ascending order")
                previous = doc.doc_id
                group_id, group = next(groups, (None, []))
                parts = list(group)
                if group_id != doc.doc_id or not parts:
                    raise ValueError("Chunks must match document order with at least one chunk per document")
                text = DocumentChunker.retrieval_text(doc)
                for index, part in enumerate(parts):
                    if part.chunk_index != index or not part.content.strip():
                        raise ValueError("Chunk indexes must be contiguous and content non-empty")
                    if not (0 <= part.char_start < part.char_end <= len(text)):
                        raise ValueError("Chunk offsets are outside parent text")
                    if text[part.char_start:part.char_end] != part.content:
                        raise ValueError("Chunk content does not match parent text")
                    if not 0 < part.token_count <= self.manifest["max_tokens_including_special"]:
                        raise ValueError("Invalid chunk token count")
                yield CorpusDocument(doc, parts)
            if next(groups, None) is not None:
                raise ValueError("Chunks contain unknown or extra documents")

    def validate(self) -> set[str]:
        ids = set()
        count = 0
        for item in self:
            ids.add(item.document.doc_id)
            count += len(item.chunks)
        if not ids or len(ids) != self.manifest["documents"] or count != self.manifest["chunks"]:
            raise ValueError("Corpus counts do not match manifest, or corpus is empty")
        self.verify_hashes()
        return ids
