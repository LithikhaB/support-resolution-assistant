"""Verify parameter boundaries and shared lexical retrieval selection."""

from app.config.settings import Settings
from app.retrieval.hybrid_search import HybridRetriever
from app.retrieval.lexical_search import PostgresLexicalRetriever, build_lexical_sql
from app.retrieval.models import RetrievalRequest


def test_shared_backend_does_not_build_an_in_memory_corpus():
    retriever = HybridRetriever(settings=Settings(_env_file=None, lexical_backend="postgres"))
    assert isinstance(retriever.bm25, PostgresLexicalRetriever)
    assert not hasattr(retriever.bm25, "_index")


def test_query_and_metadata_never_become_sql():
    marker = "x' OR 1=1 --"
    query, parameters = build_lexical_sql(
        RetrievalRequest(query=marker, filters={"queue": marker}, top_k=3)
    )
    assert marker not in query
    assert marker in parameters
    assert "plainto_tsquery" in query and "search_vector @@" in query
    assert parameters[-1] == 3
