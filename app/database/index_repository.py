"""Transactional writes for a resumable, single-writer retrieval corpus."""

import math
from pathlib import Path

from psycopg.types.json import Jsonb

from app.retrieval.corpus import CorpusDocument

DOCUMENT_UPSERT = """
INSERT INTO documents (doc_id,doc_type,title,body,response,resolution,outcome_status,
                       ticket_type,priority,intent,product,severity,sentiment,metadata)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
ON CONFLICT (doc_id) DO UPDATE SET
 doc_type=excluded.doc_type,title=excluded.title,body=excluded.body,response=excluded.response,
 resolution=excluded.resolution,outcome_status=excluded.outcome_status,ticket_type=excluded.ticket_type,
 priority=excluded.priority,intent=excluded.intent,product=excluded.product,
 severity=excluded.severity,sentiment=excluded.sentiment,metadata=excluded.metadata
"""


def vector_literal(vector: list[float]) -> str:
    """Serialize a finite normalized 384-dimensional vector for a bound SQL parameter."""
    if len(vector) != 384 or not all(math.isfinite(x) for x in vector):
        raise ValueError("Expected 384 finite embedding values")
    if not math.isclose(math.hypot(*vector), 1.0, abs_tol=1e-4):
        raise ValueError("Expected a normalized embedding")

    return "[" + ",".join(str(float(x)) for x in vector) + "]"


class IndexRepository:
    """Persist document batches and checkpoints with transactional index readiness."""

    def __init__(self, conn):
        self.conn = conn

    def migrate(self) -> None:
        """Install the indexing state tables without replacing existing documents."""
        path = Path(__file__).resolve().parents[2] / "db/migrations/003_retrieval_indexing.sql"
        with self.conn.transaction():
            self.conn.execute(path.read_text(encoding="utf-8"))

    def begin(self, config: dict, config_hash: str, manifest: dict, source_ids: set[str]) -> None:
        """Reject incompatible or incomplete source sets before marking indexing active."""
        with self.conn.transaction():
            rows = self.conn.execute(
                "SELECT doc_id,config_hash FROM document_index_state"
            ).fetchall()
            if any(row[1] != config_hash for row in rows):
                raise ValueError(
                    "Existing embeddings use a different configuration; explicit rebuild required"
                )
            if {row[0] for row in rows} - source_ids:
                raise ValueError(
                    "Input omits previously indexed documents; explicit retirement is required"
                )
            unknown = self.conn.execute("""SELECT EXISTS(SELECT 1 FROM chunks c
                WHERE embedding IS NOT NULL AND NOT EXISTS
                (SELECT 1 FROM document_index_state s WHERE s.doc_id=c.doc_id))""").fetchone()[0]
            if unknown:
                raise ValueError(
                    "Untracked embeddings exist; refusing to mix unknown model versions"
                )
            self.conn.execute(
                """INSERT INTO retrieval_index_state
                (singleton,status,config_hash,configuration,source_hash,chunks_hash,completed_at)
                VALUES (TRUE,'indexing',%s,%s,%s,%s,NULL)
                ON CONFLICT(singleton) DO UPDATE SET status='indexing',config_hash=excluded.config_hash,
                configuration=excluded.configuration,source_hash=excluded.source_hash,
                chunks_hash=excluded.chunks_hash,completed_at=NULL""",
                (config_hash, Jsonb(config), manifest["source_sha256"], manifest["output_sha256"]),
            )

    def unchanged(self, batch: list[CorpusDocument], config_hash: str) -> set[str]:
        """Identify complete documents whose content and model fingerprints still match."""
        ids = [item.document.doc_id for item in batch]
        rows = self.conn.execute(
            """SELECT s.doc_id,s.fingerprint,s.chunk_count,
                    count(c.chunk_id),count(c.embedding)
                FROM document_index_state s LEFT JOIN chunks c ON c.doc_id=s.doc_id
                WHERE s.doc_id=ANY(%s) AND s.config_hash=%s
                GROUP BY s.doc_id,s.fingerprint,s.chunk_count""",
            (ids, config_hash),
        ).fetchall()
        actual = {r[0]: r for r in rows}
        return {
            item.document.doc_id
            for item in batch
            if (row := actual.get(item.document.doc_id)) is not None
            and row[1] == item.fingerprint(config_hash)
            and row[2] == row[3] == row[4] == len(item.chunks)
        }

    def write_batch(
        self, batch: list[CorpusDocument], vectors: list[list[float]], config_hash: str
    ) -> None:
        """Atomically persist parent evidence, vectors and resume checkpoints."""
        if len(vectors) != sum(len(item.chunks) for item in batch):
            raise ValueError("Vector count does not match chunk count")
        literals = iter([vector_literal(v) for v in vectors])
        doc_rows, chunk_rows, state_rows, trim_rows = [], [], [], []
        for item in batch:
            doc = item.document.model_dump(mode="json")
            doc_rows.append(
                tuple(
                    doc.get(k)
                    for k in (
                        "doc_id",
                        "doc_type",
                        "title",
                        "body",
                        "response",
                        "resolution",
                        "outcome_status",
                        "ticket_type",
                        "priority",
                        "intent",
                        "product",
                        "severity",
                        "sentiment",
                    )
                )
                + (Jsonb(doc["metadata"]),)
            )
            for chunk in item.chunks:
                chunk_rows.append((doc["doc_id"], chunk.chunk_index, chunk.content, next(literals)))
            trim_rows.append((doc["doc_id"], len(item.chunks)))
            state_rows.append(
                (doc["doc_id"], item.fingerprint(config_hash), config_hash, len(item.chunks))
            )
        with self.conn.transaction():
            with self.conn.cursor() as cursor:
                cursor.executemany(DOCUMENT_UPSERT, doc_rows)
                cursor.executemany(
                    """INSERT INTO chunks(doc_id,chunk_index,content,embedding)
                    VALUES(%s,%s,%s,%s::vector) ON CONFLICT(doc_id,chunk_index) DO UPDATE
                    SET content=excluded.content, embedding=excluded.embedding""",
                    chunk_rows,
                )
                cursor.executemany(
                    "DELETE FROM chunks WHERE doc_id=%s AND chunk_index >= %s", trim_rows
                )
                cursor.executemany(
                    """INSERT INTO document_index_state(doc_id,fingerprint,config_hash,chunk_count)
                    VALUES(%s,%s,%s,%s) ON CONFLICT(doc_id) DO UPDATE SET
                    fingerprint=excluded.fingerprint,config_hash=excluded.config_hash,
                    chunk_count=excluded.chunk_count,indexed_at=now()""",
                    state_rows,
                )

    def finish(self, expected_documents: int, expected_chunks: int) -> None:
        """Verify counts and cosine index validity before marking the corpus ready."""
        with self.conn.transaction():
            counts = self.conn.execute("""SELECT count(DISTINCT s.doc_id),count(c.chunk_id),count(c.embedding)
                FROM document_index_state s LEFT JOIN chunks c ON c.doc_id=s.doc_id""").fetchone()
            if counts != (expected_documents, expected_chunks, expected_chunks):
                raise ValueError("Indexed counts do not match the prepared corpus")
            self.conn.execute("""CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
                ON chunks USING hnsw (embedding vector_cosine_ops)""")
            index = self.conn.execute("""SELECT i.indisvalid,pg_get_indexdef(i.indexrelid)
                FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid
                JOIN pg_namespace n ON n.oid=c.relnamespace
                WHERE c.relname='idx_chunks_embedding_hnsw' AND n.nspname=current_schema()""").fetchone()
            if (
                not index
                or not index[0]
                or "USING hnsw (embedding vector_cosine_ops)" not in index[1]
            ):
                raise ValueError("HNSW index is missing, invalid or has incompatible configuration")
            self.conn.execute("ANALYZE chunks")
            self.conn.execute(
                "UPDATE retrieval_index_state SET status='ready',completed_at=now() WHERE singleton"
            )
