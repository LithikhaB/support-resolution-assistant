"""Read-only verification of Phase D rows, embedding norms and HNSW readiness."""

import json

import psycopg

from app.database.connection import get_connection


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


def main() -> None:
    """Run the command and report its result to the terminal."""
    try:
        with get_connection(statement_timeout_ms=60000) as conn:
            report = check_index(conn)
    except psycopg.Error:
        print(
            json.dumps(
                {"status": "not_ready", "reason": "Database unavailable or indexing schema missing"}
            )
        )
        raise SystemExit(1)
    print(json.dumps(report, indent=2))
    if report["status"] != "ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
