"""Reusable token-aware chunk generation from validated local corpus artifacts."""

import json
from collections import Counter
from dataclasses import asdict

from app.ingestion.artifacts import atomic_write, file_sha256, write_json
from app.ingestion.schema import SupportDocument
from app.retrieval.chunking import DocumentChunker
from app.retrieval.tokenizer import load_tokenizer


def chunk_corpus(settings, directory=None, preview=0):
    """Regenerate chunks/manifests atomically, retaining the CLI's preview behaviour."""
    if not 0 <= preview <= 10:
        raise ValueError("preview must be between zero and ten")
    directory = directory or settings.processed_dir
    source = directory / "documents.jsonl"
    source_hash = file_sha256(source)
    parent_manifest = directory / "manifest.json"
    if parent_manifest.exists():
        manifest = json.loads(parent_manifest.read_text(encoding="utf-8"))
        if manifest.get("output_sha256") != source_hash:
            raise ValueError("Documents do not match their Day 1 manifest")
    tokenizer, tokenizer_path = load_tokenizer(
        settings.embedding_model,
        settings.tokenizer_revision,
        str(settings.data_dir / "models"),
        settings.tokenizer_local_files_only,
    )
    chunker = DocumentChunker(tokenizer, settings.chunk_max_tokens, settings.chunk_overlap_tokens)
    counts = Counter()
    lengths = []
    previews = []

    def lines():
        with source.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    document = SupportDocument.model_validate_json(line)
                except ValueError as exc:
                    raise ValueError(f"Invalid document on line {line_number}") from exc
                text = chunker.retrieval_text(document)
                lengths.append(len(chunker.tokenizer.encode(text).ids))
                chunks = chunker.chunk_document(document)
                counts["documents"] += 1
                counts["chunks"] += len(chunks)
                counts["split_documents"] += len(chunks) > 1
                for chunk in chunks:
                    data = asdict(chunk)
                    if len(previews) < preview:
                        previews.append(data)
                    yield json.dumps(data, ensure_ascii=False) + "\n"
        if not counts["documents"]:
            raise ValueError("No documents; refusing to replace chunks")
        if file_sha256(source) != source_hash:
            raise ValueError("Documents changed during chunking")

    output = directory / "chunks.jsonl"
    atomic_write(output, lines())
    lengths.sort()
    report = dict(counts) | {
        "chunking_version": "1.0.0",
        "model": settings.embedding_model,
        "tokenizer_requested_revision": settings.tokenizer_revision,
        "tokenizer_sha256": file_sha256(tokenizer_path),
        "max_tokens_including_special": chunker.max_tokens,
        "overlap_content_tokens": chunker.overlap_tokens,
        "source_sha256": source_hash,
        "output_sha256": file_sha256(output),
        "document_tokens": {
            "median": lengths[len(lengths) // 2],
            "p95": lengths[int(len(lengths) * 0.95)],
            "max": lengths[-1],
        },
        "evidence": "Join doc_id to documents.jsonl for response, resolution and outcome status",
    }
    write_json(output.with_suffix(".manifest.json"), report)
    return report, previews
