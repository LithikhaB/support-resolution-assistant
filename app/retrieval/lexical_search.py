"""Shared PostgreSQL full-text ranking for deployments with multiple API workers."""

from app.config.settings import get_settings
from app.database.connection import get_connection
from app.retrieval.models import BM25Result, RetrievalRequest
from app.retrieval.vector_search import RetrievalUnavailable


def build_lexical_sql(request):
    """Use an OR query for short complaints; bind all user text and filter values."""
    from app.retrieval.bm25_search import tokenize

    terms = sorted(set(tokenize(request.query)))
    # plainto_tsquery parses each token safely; never assemble tsquery operators from input.
    query = " || ".join("plainto_tsquery('english', %s)" for _ in terms) or "''::tsquery"
    parameters = list(terms)
    columns = {
        "doc_type": "d.doc_type",
        "intent": "d.intent",
        "severity": "d.severity",
        "ticket_type": "d.ticket_type",
        "product": "d.product",
        "queue": "d.metadata->>'queue'",
    }
    clauses = ["c.embedding IS NOT NULL", "c.search_vector @@ q.query"]
    for key, value in request.filters.model_dump(mode="json", exclude_none=True).items():
        clauses.append(f"{columns[key]} = %s")
        parameters.append(value)
    parameters.append(request.top_k)
    sql = f"""WITH q AS (SELECT {query} AS query)
        SELECT c.chunk_id,c.doc_id,c.chunk_index,c.content,d.title,d.doc_type,
            d.response,d.resolution,d.outcome_status,d.metadata,
            ts_rank_cd(c.search_vector,q.query) AS score
        FROM chunks c JOIN documents d ON d.doc_id=c.doc_id CROSS JOIN q
        WHERE {" AND ".join(clauses)} ORDER BY score DESC,c.chunk_id LIMIT %s"""
    return sql, parameters


class PostgresLexicalRetriever:
    """Rank in the database without loading a corpus copy into each API process."""

    def __init__(self, *, connection_factory=get_connection, settings=None):
        self.connection_factory = connection_factory
        self.settings = settings or get_settings()

    def search(self, query, top_k=10, filters=None):
        request = RetrievalRequest(query=query, top_k=top_k, filters=filters or {})
        with self.connection_factory() as conn:
            return self.search_connection(conn, request)

    def search_connection(self, conn, request):
        conn.execute(
            "SELECT set_config('statement_timeout',%s,true)",
            (str(self.settings.retrieval_statement_timeout_ms),),
        )
        if not conn.execute("SELECT pg_try_advisory_xact_lock_shared(8041,2)").fetchone()[0]:
            raise RetrievalUnavailable("Indexing is running; retry after it completes")
        state = conn.execute("SELECT status FROM retrieval_index_state WHERE singleton").fetchone()
        if not state or state[0] != "ready":
            raise RetrievalUnavailable("Retrieval index is not ready")
        sql, parameters = build_lexical_sql(request)
        rows = conn.execute(sql, parameters).fetchall()
        return [
            BM25Result(
                chunk_id=row[0],
                doc_id=row[1],
                chunk_index=row[2],
                content=row[3],
                title=row[4],
                doc_type=row[5],
                response=row[6],
                resolution=row[7],
                outcome_status=row[8],
                metadata=row[9],
                bm25_score=row[10],
                bm25_rank=rank,
            )
            for rank, row in enumerate(rows, 1)
        ]
