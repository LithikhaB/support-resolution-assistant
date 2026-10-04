"""Inspect the persisted retrieval index without loading model weights."""


def check_index(conn) -> dict:
    """Inspect corpus counts, embedding validity and the cosine HNSW index."""
    state = conn.execute("SELECT status FROM retrieval_index_state WHERE singleton").fetchone()
    documents, expected_chunks = conn.execute(
        "SELECT count(*),coalesce(sum(chunk_count),0) FROM document_index_state"
    ).fetchone()
    chunks, embeddings, invalid = conn.execute("""SELECT count(*),count(embedding),
        count(*) FILTER (WHERE embedding IS NOT NULL AND
        (vector_dims(embedding) <> 384 OR abs(1 + (embedding <#> embedding)) > 0.0002))
        FROM chunks""").fetchone()
    index = conn.execute("""SELECT i.indisvalid,pg_get_indexdef(i.indexrelid)
        FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid
        JOIN pg_namespace n ON n.oid=c.relnamespace
        WHERE c.relname='idx_chunks_embedding_hnsw' AND n.nspname=current_schema()""").fetchone()
    valid_index = bool(
        index and index[0] and "USING hnsw (embedding vector_cosine_ops)" in index[1]
    )
    ready = bool(
        state
        and state[0] == "ready"
        and documents > 0
        and chunks > 0
        and chunks == expected_chunks == embeddings
        and invalid == 0
        and valid_index
    )
    return {
        "status": "ready" if ready else "not_ready",
        "documents_indexed": documents,
        "chunks": chunks,
        "embeddings": embeddings,
        "invalid_embeddings": invalid,
        "cosine_hnsw_valid": valid_index,
    }


def index_matches(conn, source_hash: str, chunks_hash: str) -> bool:
    """Require a published index built from the exact expected corpus artifacts."""
    row = conn.execute(
        "SELECT status,source_hash,chunks_hash FROM retrieval_index_state WHERE singleton"
    ).fetchone()
    return bool(row and tuple(row) == ("ready", source_hash, chunks_hash))
