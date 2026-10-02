import os
from pathlib import Path

import psycopg
import pytest

from test_indexing_db import repository, vectors
from app.ingestion.schema import SupportDocument
from app.ingestion.synthetic import build_dataset, load_scenarios
from app.retrieval.chunking import Chunk
from app.retrieval.corpus import CorpusDocument

pytestmark = pytest.mark.skipif(os.environ.get('RUN_DB_TESTS') != '1', reason='Set RUN_DB_TESTS=1')


def test_simulated_outcome_roundtrip_and_migration(repository):
    migration = Path('db/migrations/004_synthetic_outcomes.sql').read_text()
    repository.conn.execute(migration)
    repository.conn.execute(migration)
    document = SupportDocument.model_validate(build_dataset(load_scenarios())['tickets'][0])
    record = CorpusDocument(document, [Chunk(document.doc_id, 0, document.body, 50, 0, len(document.body))])
    repository.begin({'model': 'test'}, 'config', {'source_sha256': 'docs', 'output_sha256': 'chunks'}, {document.doc_id})
    repository.write_batch([record], vectors(1), 'config')
    status = repository.conn.execute('SELECT outcome_status FROM documents').fetchone()[0]
    assert status == 'simulated_resolved'
    with pytest.raises(psycopg.errors.CheckViolation):
        with repository.conn.transaction():
            repository.conn.execute("UPDATE documents SET outcome_status='verified_resolved'")
