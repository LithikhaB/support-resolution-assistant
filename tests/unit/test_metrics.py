"""Verify health counters without exposing URLs or customer messages."""

from fastapi.testclient import TestClient

from app.main import app


def test_metrics_count_errors_without_retaining_private_path():
    client = TestClient(app)
    before = client.get("/api/v1/metrics").json()
    client.get("/private-customer-marker")
    after = client.get("/api/v1/metrics")
    assert after.json()["client_errors"] == before["client_errors"] + 1
    assert after.json()["requests"] == before["requests"] + 1
    assert after.json()["in_flight"] == 0
    assert "private-customer-marker" not in after.text
