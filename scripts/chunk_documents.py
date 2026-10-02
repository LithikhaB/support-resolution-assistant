"""Create token-aware chunks without embeddings or database writes."""
import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from app.config.settings import get_settings
from app.ingestion.artifacts import atomic_write, file_sha256, write_json
from app.ingestion.schema import SupportDocument
from app.retrieval.chunking import DocumentChunker
from app.retrieval.tokenizer import load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", type=int, default=0, help="Print the first N chunks; explicit opt-in to displaying source text")
    parser.add_argument("--directory", type=Path, help="Stage a separate processed corpus; defaults to DATA_DIR/processed")
    args = parser.parse_args()
    if not 0 <= args.preview <= 10:
        parser.error("--preview must be between 0 and 10")
    settings = get_settings()
    directory = args.directory or settings.processed_dir
    source = directory / "documents.jsonl"
    source_hash = file_sha256(source)
    parent_manifest = directory / "manifest.json"
    if parent_manifest.exists():
        manifest = json.loads(parent_manifest.read_text(encoding="utf-8"))
        if manifest.get("output_sha256") != source_hash:
            raise ValueError("Documents do not match their Day 1 manifest")
    tokenizer, tokenizer_path = load_tokenizer(
        settings.embedding_model, settings.tokenizer_revision,
        str(settings.data_dir / "models"), settings.tokenizer_local_files_only)
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
                    if len(previews) < args.preview:
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
        "chunking_version": "1.0.0", "model": settings.embedding_model,
        "tokenizer_requested_revision": settings.tokenizer_revision,
        "tokenizer_sha256": file_sha256(tokenizer_path),
        "max_tokens_including_special": chunker.max_tokens,
        "overlap_content_tokens": chunker.overlap_tokens,
        "source_sha256": source_hash, "output_sha256": file_sha256(output),
        "document_tokens": {"median": lengths[len(lengths)//2], "p95": lengths[int(len(lengths)*.95)], "max": lengths[-1]},
        "evidence": "Join doc_id to documents.jsonl for response, resolution and outcome status",
    }
    write_json(output.with_suffix(".manifest.json"), report)
    print(json.dumps(report, indent=2))
    for chunk in previews:
        print(json.dumps(chunk, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
