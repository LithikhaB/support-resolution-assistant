"""Database cosine retrieval. Filtered queries use exact search for full coverage."""

import logging
from time import perf_counter

from app.config.settings import get_settings
from app.database.connection import get_connection
from app.database.index_repository import vector_literal
from app.retrieval.embeddings import get_embedding_service
from app.retrieval.models import RetrievalFilters, RetrievalRequest, VectorResult

logger = logging.getLogger(__name__)


class RetrievalUnavailable(RuntimeError):
    """The corpus is incomplete, incompatible, or currently being updated."""


def build_search_sql(filters: RetrievalFilters, vector: str, top_k: int):
    """Build parameterized cosine search with exact filtering before evidence loading."""
    clauses = ["c.embedding IS NOT NULL"]
    parameters = [vector]
    columns = {
        "doc_type": "d.doc_type",
        "intent": "d.intent",
        "severity": "d.severity",
        "ticket_type": "d.ticket_type",
        "product": "d.product",
        "queue": "d.metadata->>'queue'",
    }
    for key, value in filters.model_dump(mode="json", exclude_none=True).items():
        clauses.append(f"{columns[key]} = %s")
        parameters.append(value)

    filtered = any(value is not None for value in filters.model_dump().values())
    query = """SELECT c.chunk_id,c.doc_id,c.chunk_index,c.content,d.title,d.doc_type,
        d.response,d.resolution,d.outcome_status,d.metadata,
        c.embedding <=> %s::vector AS distance
        FROM chunks c JOIN documents d ON d.doc_id=c.doc_id WHERE """ + " AND ".join(clauses)
    if filtered:
        query = (
            """WITH eligible AS MATERIALIZED (
            SELECT c.chunk_id, c.embedding <=> %s::vector AS distance
            FROM chunks c JOIN documents d ON d.doc_id=c.doc_id WHERE """
            + " AND ".join(clauses)
            + """
            ), winners AS (SELECT * FROM eligible ORDER BY distance,chunk_id LIMIT %s)
            SELECT c.chunk_id,c.doc_id,c.chunk_index,c.content,d.title,d.doc_type,
                d.response,d.resolution,d.outcome_status,d.metadata,w.distance
            FROM winners w JOIN chunks c ON c.chunk_id=w.chunk_id
            JOIN documents d ON d.doc_id=c.doc_id ORDER BY w.distance,c.chunk_id"""
        )
    else:
        query += " ORDER BY c.embedding <=> %s::vector LIMIT %s"
        parameters.append(vector)
    parameters.append(top_k)
    return query, parameters


class VectorRetriever:
    """Retrieve compatible pgvector evidence under the indexing guard."""

    def __init__(self, *, embedder=None, connection_factory=get_connection, settings=None):
        self.embedder = embedder
        self.connection_factory = connection_factory
        self.settings = settings or get_settings()

    def search(
        self, query: str, top_k: int = 10, filters: RetrievalFilters | dict | None = None
    ) -> list[VectorResult]:
        """Return bounded ranked evidence for a validated query and its filters."""
        request = RetrievalRequest(
            query=query, top_k=top_k, filters={} if filters is None else filters
        )
        started = perf_counter()
        with self.connection_factory() as conn:
            conn.execute(
                "SELECT set_config('statement_timeout',%s,true)",
                (str(self.settings.retrieval_statement_timeout_ms),),
            )

            if not conn.execute("SELECT pg_try_advisory_xact_lock_shared(8041,2)").fetchone()[0]:
                raise RetrievalUnavailable("Indexing is running; retry after it completes")
            state = conn.execute(
                "SELECT status,configuration FROM retrieval_index_state WHERE singleton"
            ).fetchone()
            if not state or state[0] != "ready":
                raise RetrievalUnavailable(
                    "Retrieval index is not ready; run scripts.index_documents"
                )
            config = state[1]
            expected = {
                "model": self.settings.embedding_model,
                "tokenizer_requested_revision": self.settings.tokenizer_revision,
                "dimension": self.settings.embedding_dim,
                "normalized": True,
            }
            if any(config.get(key) != value for key, value in expected.items()):
                raise RetrievalUnavailable(
                    "Query embedding configuration differs from the indexed corpus"
                )
            embedder = self.embedder or get_embedding_service()
            vector = vector_literal(embedder.embed_query(request.query))
            sql, parameters = build_search_sql(request.filters, vector, request.top_k)

            conn.execute(
                "SELECT set_config('hnsw.ef_search',%s,true)", (str(max(100, request.top_k)),)
            )
            rows = conn.execute(sql, parameters).fetchall()
        results = [
            VectorResult(
                chunk_id=r[0],
                doc_id=r[1],
                chunk_index=r[2],
                content=r[3],
                title=r[4],
                doc_type=r[5],
                response=r[6],
                resolution=r[7],
                outcome_status=r[8],
                metadata=r[9],
                cosine_distance=r[10],
                cosine_similarity=1 - r[10],
                vector_rank=i,
            )
            for i, r in enumerate(rows, 1)
        ]
        logger.info(
            "Vector retrieval query_chars=%d top_k=%d results=%d elapsed_ms=%.1f",
            len(request.query),
            request.top_k,
            len(results),
            (perf_counter() - started) * 1000,
        )
        return results
