"""Reject incomplete retrieval indexes and mismatched evaluation artifacts."""

from unittest.mock import Mock

import pytest

from app.database.readiness import check_index, index_matches


@pytest.mark.parametrize(
    "chunks,embeddings,invalid,index_valid,ready",
    [
        (0, 0, 0, True, False),
        (2, 1, 0, True, False),
        (2, 2, 1, True, False),
        (2, 2, 0, False, False),
        (2, 2, 0, True, True),
    ],
)
def test_index_audit_rejects_incomplete_vectors(chunks, embeddings, invalid, index_valid, ready):
    """Counts alone cannot make an empty, malformed or unindexed corpus ready."""
    conn = Mock()
    conn.execute.return_value.fetchone.side_effect = [
        ("ready",),
        (1, chunks),
        (chunks, embeddings, invalid),
        (index_valid, "CREATE INDEX USING hnsw (embedding vector_cosine_ops)"),
    ]
    assert (check_index(conn)["status"] == "ready") is ready


@pytest.mark.parametrize(
    "row,expected",
    [
        (None, False),
        (("building", "a", "b"), False),
        (("ready", "old", "b"), False),
        (("ready", "a", "old"), False),
        (("ready", "a", "b"), True),
    ],
)
def test_index_fingerprint_requires_both_artifacts(row, expected):
    """Evaluation requires the exact corpus and chunk artifacts it has frozen."""
    conn = Mock()
    conn.execute.return_value.fetchone.return_value = row
    assert index_matches(conn, "a", "b") is expected
