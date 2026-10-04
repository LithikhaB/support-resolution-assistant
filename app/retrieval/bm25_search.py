"""In-memory BM25 over the indexed chunk corpus, refreshed after reindexing.

Uses k1=1.5, b=0.75 and positive Robertson IDF log(1+(N-df+.5)/(df+.5)).
Complaint and KB chunks are scored; ticket responses remain parent evidence.
"""

import re
from collections import Counter, defaultdict
from math import log1p
from threading import Lock

from app.config.settings import get_settings
from app.database.connection import get_connection
from app.retrieval.models import BM25Result, EvidenceResult, RetrievalRequest
from app.retrieval.vector_search import RetrievalUnavailable


def tokenize(text):
    """Normalize words while preserving technical identifiers and dotted versions."""
    return re.findall(r"\w+(?:[-./]\w+)*", text.casefold())


class BM25Index:
    """Score indexed complaint and KB text using a compact inverted BM25 index."""

    def __init__(self, records):
        self.records = records
        self.postings = defaultdict(list)
        self.lengths = []
        for index, record in enumerate(records):
            terms = Counter(tokenize(record["content"]))
            self.lengths.append(sum(terms.values()))
            for term, count in terms.items():
                self.postings[term].append((index, count))
        self.average_length = sum(self.lengths) / max(1, len(records))

    def search(self, request):
        """Return bounded ranked evidence for a validated query and its filters."""
        scores = defaultdict(float)
        filters = request.filters.model_dump(mode="json", exclude_none=True)
        for term in sorted(set(tokenize(request.query))):
            postings = self.postings.get(term, ())
            idf = log1p((len(self.records) - len(postings) + 0.5) / (len(postings) + 0.5))
            for index, frequency in postings:
                record = self.records[index]
                if any(
                    (record["metadata"].get(key) if key == "queue" else record.get(key)) != value
                    for key, value in filters.items()
                ):
                    continue
                norm = 1.5 * (0.25 + 0.75 * self.lengths[index] / self.average_length)
                scores[index] += idf * frequency * 2.5 / (frequency + norm)
        winners = sorted(scores, key=lambda i: (-scores[i], self.records[i]["chunk_id"]))[
            : request.top_k
        ]
        return [
            BM25Result(
                **EvidenceResult.model_validate(self.records[i]).model_dump(),
                bm25_score=scores[i],
                bm25_rank=rank,
            )
            for rank, i in enumerate(winners, 1)
        ]


class BM25Retriever:
    """Refresh the lexical cache when the guarded database corpus changes."""

    def __init__(self, *, connection_factory=get_connection, settings=None):
        self.connection_factory = connection_factory
        self.settings = settings or get_settings()
        self._lock = Lock()
        self._signature = None
        self._index = None

    def search(self, query, top_k=10, filters=None):
        """Return bounded ranked evidence for a validated query and its filters."""
        request = RetrievalRequest(
            query=query, top_k=top_k, filters={} if filters is None else filters
        )
        with self.connection_factory() as conn:
            return self.search_connection(conn, request)

    def search_connection(self, conn, request):
        """Load or reuse BM25 state within the caller's guarded transaction."""
        conn.execute(
            "SELECT set_config('statement_timeout',%s,true)",
            (str(self.settings.retrieval_statement_timeout_ms),),
        )
        if not conn.execute("SELECT pg_try_advisory_xact_lock_shared(8041,2)").fetchone()[0]:
            raise RetrievalUnavailable("Indexing is running; retry after it completes")
        state = conn.execute("""SELECT status,config_hash,source_hash,chunks_hash,completed_at
                                FROM retrieval_index_state WHERE singleton""").fetchone()
        if not state or state[0] != "ready":
            raise RetrievalUnavailable("Retrieval index is not ready; run scripts.index_documents")
        signature = tuple(state[1:])
        with self._lock:
            if self._index is None or signature != self._signature:
                rows = conn.execute("""SELECT c.chunk_id,c.doc_id,c.chunk_index,c.content,
                    d.title,d.doc_type,d.response,d.resolution,d.outcome_status,d.metadata,
                    d.intent,d.severity,d.ticket_type,d.product,d.body
                    FROM chunks c JOIN documents d ON d.doc_id=c.doc_id
                    WHERE c.embedding IS NOT NULL ORDER BY c.chunk_id""").fetchall()
                fields = (
                    "chunk_id",
                    "doc_id",
                    "chunk_index",
                    "content",
                    "title",
                    "doc_type",
                    "response",
                    "resolution",
                    "outcome_status",
                    "metadata",
                    "intent",
                    "severity",
                    "ticket_type",
                    "product",
                    "evidence_content",
                )
                self._index = BM25Index([dict(zip(fields, row)) for row in rows])
                self._signature = signature
            index = self._index
        return index.search(request)
