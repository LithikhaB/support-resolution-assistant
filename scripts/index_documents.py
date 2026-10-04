"""Index verified local artifacts in restartable batches; never reset the database."""

import argparse
import json
import logging
from importlib.metadata import version
from itertools import batched
from time import perf_counter

from app.config.settings import get_settings
from app.database.connection import get_connection
from app.database.index_repository import IndexRepository
from app.ingestion.artifacts import digest, file_sha256, write_json
from app.monitoring.logging import configure_logging
from app.retrieval.corpus import PreparedCorpus
from app.retrieval.embeddings import get_embedding_service
from app.retrieval.tokenizer import load_tokenizer

logger = logging.getLogger(__name__)


def indexing_configuration(settings, manifest: dict) -> dict:
    """Verify tokenizer compatibility and fingerprint the embedding configuration."""
    expected = {
        "model": settings.embedding_model,
        "tokenizer_requested_revision": settings.tokenizer_revision,
        "max_tokens_including_special": settings.chunk_max_tokens,
        "overlap_content_tokens": settings.chunk_overlap_tokens,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Chunk manifest differs from current model/chunk settings; rerun chunking")
    _, path = load_tokenizer(
        settings.embedding_model,
        settings.tokenizer_revision,
        str(settings.data_dir / "models"),
        settings.embedding_local_files_only,
    )
    if file_sha256(path) != manifest["tokenizer_sha256"]:
        raise ValueError("Tokenizer differs from the one used for chunking")
    if settings.embedding_dim != 384:
        raise ValueError("Database requires 384 dimensions")
    return expected | {
        "dimension": 384,
        "normalized": True,
        "device": "cpu",
        "tokenizer_sha256": manifest["tokenizer_sha256"],
        "chunking_version": manifest["chunking_version"],
        "embedding_contract": 1,
        "sentence_transformers": version("sentence-transformers"),
        "torch": version("torch"),
        "tokenizers": version("tokenizers"),
    }


def main() -> None:
    """Run the command and report its result to the terminal."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Re-embed tracked documents when the runtime configuration changes; preserve source rows and resume completed batches.",
    )
    args = parser.parse_args()
    configure_logging(settings.log_level)
    started = perf_counter()
    corpus = PreparedCorpus(settings.processed_dir)
    logger.info("Validating prepared corpus before database writes")
    ids = corpus.validate()
    config = indexing_configuration(settings, corpus.manifest)
    config_hash = digest(config)
    stats = {
        "documents_processed": 0,
        "documents_upserted": 0,
        "documents_skipped": 0,
        "chunks_upserted": 0,
        "embeddings_generated": 0,
        "batches_written": 0,
    }
    with get_connection(statement_timeout_ms=settings.indexing_statement_timeout_ms) as conn:
        conn.autocommit = True
        if not conn.execute("SELECT pg_try_advisory_lock(8041, 2)").fetchone()[0]:
            raise RuntimeError("Another indexing process is running")
        try:
            repo = IndexRepository(conn)
            repo.migrate()
            repo.begin(config, config_hash, corpus.manifest, ids, rebuild=args.rebuild)
            for group in batched(corpus, settings.indexing_batch_size):
                batch = list(group)
                unchanged = repo.unchanged(batch, config_hash)
                changed = [item for item in batch if item.document.doc_id not in unchanged]
                if changed:
                    texts = [chunk.content for item in changed for chunk in item.chunks]
                    vectors = get_embedding_service().embed_documents(texts)
                    repo.write_batch(changed, vectors, config_hash)
                    stats["documents_upserted"] += len(changed)
                    stats["chunks_upserted"] += len(texts)
                    stats["embeddings_generated"] += len(vectors)
                    stats["batches_written"] += 1
                stats["documents_skipped"] += len(unchanged)
                stats["documents_processed"] += len(batch)
                logger.info(
                    "Indexing progress processed=%d/%d upserted=%d skipped=%d",
                    stats["documents_processed"],
                    len(ids),
                    stats["documents_upserted"],
                    stats["documents_skipped"],
                )
            corpus.verify_hashes()
            logger.info("Validating row counts and creating cosine HNSW index")
            repo.finish(len(ids), corpus.manifest["chunks"])
        finally:
            conn.execute("SELECT pg_advisory_unlock(8041, 2)")
    report = stats | {
        "status": "ready",
        "elapsed_seconds": round(perf_counter() - started, 2),
        "source_sha256": corpus.manifest["source_sha256"],
        "chunks_sha256": corpus.manifest["output_sha256"],
        "configuration": config,
        "config_hash": config_hash,
        "index": "idx_chunks_embedding_hnsw",
    }
    write_json(settings.processed_dir / "indexing.manifest.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
