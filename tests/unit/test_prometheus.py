"""Verify content-free bounded counters and cumulative stage histograms."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from app.monitoring import metrics


@pytest.fixture
def isolated(monkeypatch):
    monkeypatch.setattr(metrics, "_provider_counts", Counter())
    monkeypatch.setattr(metrics, "_citation_counts", Counter())
    monkeypatch.setattr(metrics, "_cache_counts", Counter())
    monkeypatch.setattr(metrics, "_fallback_count", 0)
    monkeypatch.setattr(
        metrics,
        "_stage_histograms",
        {
            stage: {"buckets": [0] * len(metrics._buckets), "sum": 0.0, "count": 0}
            for stage in ("understand", "retrieve", "draft", "validate")
        },
    )


def test_prometheus_labels_are_bounded_and_do_not_expose_model_or_error(isolated):
    from app.llm.telemetry import record

    record("groq", "private-model", error="provider_http_429")
    metrics.provider_outcome("customer@example.com", "private error")
    metrics.solution_cache(True)
    metrics.solution_cache(False)
    metrics.citation_validation("passed")
    metrics.resolution_fallback()
    body = metrics.prometheus_metrics().body.decode()
    assert 'resolve_provider_total{provider="groq",outcome="throttled"} 1' in body
    assert 'resolve_provider_total{provider="other",outcome="error"} 1' in body
    assert 'citation_validation_total{status="passed"} 1' in body
    assert 'solution_cache_total{outcome="hit"} 1' in body
    assert 'solution_cache_total{outcome="miss"} 1' in body
    assert "resolve_fallback_total 1" in body
    assert "private" not in body and "@" not in body and "429" not in body


def test_stage_histogram_counts_failures_and_preserves_returns(isolated, monkeypatch):
    ticks = iter((0.0, 0.3, 1.0, 1.7))
    monkeypatch.setattr(metrics, "perf_counter", lambda: next(ticks))
    response = SimpleNamespace(validation=SimpleNamespace(status="passed"))
    assert metrics.timed_stage("validate", lambda: response) is response
    with pytest.raises(RuntimeError, match="same exception"):
        with metrics.stage_duration("retrieve"):
            raise RuntimeError("same exception")
    body = metrics.prometheus_metrics().body.decode()
    assert 'stage_duration_seconds_bucket{stage="validate",le="0.1"} 0' in body
    assert 'stage_duration_seconds_bucket{stage="validate",le="0.5"} 1' in body
    assert 'stage_duration_seconds_bucket{stage="retrieve",le="0.5"} 0' in body
    assert 'stage_duration_seconds_bucket{stage="retrieve",le="1"} 1' in body
    assert 'stage_duration_seconds_bucket{stage="retrieve",le="+Inf"} 1' in body
    assert 'stage_duration_seconds_count{stage="validate"} 1' in body
    assert 'citation_validation_total{status="passed"} 1' in body


def test_metric_updates_are_thread_safe_and_missing_keys_are_counted(isolated):
    from app.config.settings import Settings
    from app.llm.client import GroqClient, LanguageUnavailable

    with ThreadPoolExecutor(max_workers=20) as pool:
        list(pool.map(lambda _: metrics.solution_cache(True), range(1000)))
    assert metrics._cache_counts["hit"] == 1000
    client = GroqClient(Settings(_env_file=None))
    with pytest.raises(LanguageUnavailable, match="missing_api_key"):
        client.generate("test", {}, None)
    assert metrics._provider_counts["groq", "missing_key"] == 1
