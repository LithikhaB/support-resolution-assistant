from contextlib import contextmanager
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from app.config.settings import Settings
from app.retrieval.models import RetrievalFilters, RetrievalRequest
from app.retrieval.vector_search import VectorRetriever, RetrievalUnavailable, build_search_sql


@pytest.mark.parametrize('values',[{'query':' '},{'query':'ok','top_k':0},
    {'query':'ok','top_k':101},{'query':'ok','top_k':True},
    {'query':'ok','filters':{'sql':'bad'}},{'query':'ok','filters':{'queue':' '}}])
def test_invalid_requests(values):
    with pytest.raises(ValidationError): RetrievalRequest(**values)


def test_filter_values_are_parameters():
    attack="x' OR 1=1 --"
    sql,params=build_search_sql(RetrievalFilters(queue=attack),'[1]',5)
    assert attack not in sql and attack in params
    assert 'MATERIALIZED' in sql
    assert 'c.content' not in sql.split('), winners')[0]
    assert params[-1]==5


def setup_retriever(state='ready',lock=True):
    settings=Settings(_env_file=None)
    config={'model':settings.embedding_model,'tokenizer_requested_revision':settings.tokenizer_revision,
            'dimension':384,'normalized':True}
    conn=Mock()
    def execute(sql,params=None):
        result=Mock()
        if 'advisory' in sql: result.fetchone.return_value=(lock,)
        elif 'retrieval_index_state' in sql: result.fetchone.return_value=(state,config)
        else: result.fetchall.return_value=[(1,'doc',0,'complaint','Title','historical_response',
                                            'Waiting for details',None,'unknown',{},0.25)]
        return result
    conn.execute.side_effect=execute
    @contextmanager
    def connection(): yield conn
    embedder=Mock()
    embedder.embed_query.return_value=[1.0]+[0.0]*383
    return VectorRetriever(embedder=embedder,connection_factory=connection,settings=settings),embedder,config


def test_results_preserve_evidence_and_distance_semantics():
    retriever,embedder,_=setup_retriever()
    result=retriever.search(' complaint ')[0]
    assert result.cosine_distance==0.25 and result.cosine_similarity==0.75
    assert result.response=='Waiting for details' and result.resolution is None
    assert result.outcome_status=='unknown' and result.vector_rank==1
    embedder.embed_query.assert_called_once_with('complaint')


@pytest.mark.parametrize('state,lock',[('indexing',True),('ready',False)])
def test_not_ready_rejected_before_embedding(state,lock):
    retriever,embedder,_=setup_retriever(state,lock)
    with pytest.raises(RetrievalUnavailable): retriever.search('complaint')
    embedder.embed_query.assert_not_called()


def test_model_mismatch_rejected():
    retriever,embedder,config=setup_retriever()
    config['model']='other'
    with pytest.raises(RetrievalUnavailable): retriever.search('complaint')
    embedder.embed_query.assert_not_called()
