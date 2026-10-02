import os
from contextlib import contextmanager
from unittest.mock import Mock

import pytest

from test_indexing_db import repository, item, vectors
from app.config.settings import Settings
from app.retrieval.vector_search import VectorRetriever
from app.retrieval.bm25_search import BM25Retriever
from app.retrieval.hybrid_search import HybridRetriever

pytestmark=pytest.mark.skipif(os.environ.get('RUN_DB_TESTS')!='1',reason='Set RUN_DB_TESTS=1')


def test_database_ranking_filters_and_evidence(repository):
    settings=Settings(_env_file=None)
    config={'model':settings.embedding_model,'tokenizer_requested_revision':settings.tokenizer_revision,
            'dimension':384,'normalized':True}
    record=item()
    repository.begin(config,'config',{'source_sha256':'docs','output_sha256':'chunks'},{'test'})
    repository.write_batch([record],vectors(1),'config')
    repository.finish(1,1)
    @contextmanager
    def connection(): yield repository.conn
    embedder=Mock()
    embedder.embed_query.return_value=vectors(1)[0]
    retriever=VectorRetriever(settings=settings,embedder=embedder,connection_factory=connection)
    results=retriever.search('network issue',3)
    assert len(results)==1 and results[0].cosine_distance==pytest.approx(0)
    assert results[0].cosine_similarity==pytest.approx(1)
    assert results[0].response==record.document.response
    assert retriever.search('network issue',3,{'doc_type':'knowledge_base'})==[]
    assert len(retriever.search('network issue',3,{'doc_type':'historical_response'}))==1
    assert retriever.search('network issue',3,{'queue':"x' OR 1=1 --"})==[]
    # Both branches must read the same generation and preserve original evidence.
    keyword = BM25Retriever(settings=settings,connection_factory=connection)
    query = record.chunks[0].content
    lexical = keyword.search(query,3)
    assert len(lexical)==1 and lexical[0].response==record.document.response
    assert keyword.search(query,3,{'doc_type':'knowledge_base'})==[]
    hybrid = HybridRetriever(settings=settings,embedder=embedder,connection_factory=connection)
    combined = hybrid.search(query,3)
    assert len(combined)==1 and combined[0].sources==['vector','bm25']
    assert combined[0].rrf_score==pytest.approx(2/61)
    assert hybrid.search(query,3,{'queue':"x' OR 1=1 --"})==[]
