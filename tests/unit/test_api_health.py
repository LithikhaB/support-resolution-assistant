from contextlib import contextmanager
from unittest.mock import MagicMock

import psycopg
from fastapi.testclient import TestClient

from app.api import routes
from app.main import app


def test_health():
    resp = TestClient(app).get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_readiness_failure_does_not_break_liveness(monkeypatch):
    @contextmanager
    def unavailable():
        raise psycopg.OperationalError("private connection information")
        yield

    monkeypatch.setattr(routes, "get_connection", unavailable)
    client = TestClient(app)
    assert client.get("/api/v1/health").status_code == 200
    response = client.get("/api/v1/ready")
    assert response.status_code == 503
    assert "private" not in response.text


def test_readiness_checks_schema(monkeypatch):
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value

    @contextmanager
    def connection():
        yield conn

    monkeypatch.setattr(routes, "get_connection", connection)
    client = TestClient(app)
    cursor.fetchone.return_value = (True, True, True, False)
    assert client.get("/api/v1/ready").status_code == 503
    cursor.fetchone.return_value = (True, True, True, True)
    assert client.get("/api/v1/ready").json() == {"status": "ready", "scope": "database_schema"}
