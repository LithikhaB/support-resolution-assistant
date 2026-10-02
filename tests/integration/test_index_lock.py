"""Check writer/reader exclusion using genuinely separate database sessions."""

import os
from unittest.mock import Mock

import pytest

from app.database.connection import get_connection
from app.retrieval.bm25_search import BM25Retriever
from app.retrieval.vector_search import RetrievalUnavailable, VectorRetriever

pytestmark = pytest.mark.skipif(os.environ.get("RUN_DB_TESTS") != "1", reason="Set RUN_DB_TESTS=1")


def test_active_indexer_blocks_both_retrievers_before_embedding():
    embedder = Mock()
    with get_connection() as writer:
        assert writer.execute("SELECT pg_try_advisory_xact_lock(8041,2)").fetchone()[0]
        for retriever in (BM25Retriever(), VectorRetriever(embedder=embedder)):
            with pytest.raises(RetrievalUnavailable, match="Indexing"):
                retriever.search("router")
    embedder.embed_query.assert_not_called()
