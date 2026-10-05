"""Verify load percentiles include rejected requests and permit concurrency 20."""

import json
import sys

import pytest

from scripts import load_test


def test_load_report_has_tail_latency_rps_and_errors(monkeypatch, tmp_path):
    output = tmp_path / "load.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["load_test", "--requests", "2", "--concurrency", "20", "--output", str(output)],
    )
    monkeypatch.setattr(load_test, "measure", lambda *args: {"status": 503, "ms": 12.0})
    ticks = iter((1.0, 3.0))
    monkeypatch.setattr(load_test, "perf_counter", lambda: next(ticks))
    load_test.main()
    report = json.loads(output.read_text())
    assert report["concurrency"] == 20
    assert report["warm_p50_ms"] == report["warm_p95_ms"] == report["warm_p99_ms"] == 12
    assert report["completed_requests_per_second"] == 1
    assert report["successful_requests_per_second"] == 0
    assert report["error_rate"] == 1
    assert report["successful_p95_ms"] is None
    with pytest.raises(SystemExit):
        load_test.main()
