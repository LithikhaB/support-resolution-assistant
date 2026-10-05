"""Collect bounded process-local request health without storing customer content."""

from collections import Counter, deque
from contextlib import contextmanager
from hashlib import sha256
from threading import Lock
from time import perf_counter

import psycopg
from fastapi import APIRouter
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.concurrency import run_in_threadpool

from app.config.settings import get_settings
from app.database.connection import pool_stats
from app.llm.telemetry import snapshot as language_metrics
from app.monitoring.budgets import api_budget

router = APIRouter(prefix="/api/v1")
_lock = Lock()
_durations = deque(maxlen=1000)
_counts = {
    "requests": 0,
    "server_errors": 0,
    "client_errors": 0,
    "in_flight": 0,
    "throttled": 0,
    "capacity_rejected": 0,
}
_resolution_active = 0
_buckets = (0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 120)
_histogram = [0] * len(_buckets)
_duration_sum = 0.0


_provider_counts = Counter()
_citation_counts = Counter()
_cache_counts = Counter()
_fallback_count = 0
_stage_histograms = {
    stage: {"buckets": [0] * len(_buckets), "sum": 0.0, "count": 0}
    for stage in ("understand", "retrieve", "draft", "validate")
}


def provider_outcome(provider, outcome):
    """Count provider attempts with bounded labels; exclude prompts and error strings."""
    provider = provider if provider in {"groq", "gemini", "local"} else "other"
    outcome = (
        outcome
        if outcome
        in {
            "success",
            "error",
            "throttled",
            "missing_key",
            "cache_hit",
            "circuit_open",
            "extractive",
        }
        else "error"
    )
    with _lock:
        _provider_counts[provider, outcome] += 1


def solution_cache(hit):
    """Count an enabled persistent evidence-cache lookup, not downstream model caches."""
    with _lock:
        _cache_counts["hit" if hit else "miss"] += 1


def citation_validation(status):
    """Count each citation validator invocation, including repeated final validation."""
    status = status if status in {"passed", "failed", "not_run"} else "other"
    with _lock:
        _citation_counts[status] += 1


def resolution_fallback():
    """Count a resolution using fallback after attempted language drafting fails."""
    global _fallback_count
    with _lock:
        _fallback_count += 1


@contextmanager
def stage_duration(stage):
    """Observe stage operations, including failures; a request can invoke a stage twice."""
    started = perf_counter()
    try:
        yield
    finally:
        seconds = perf_counter() - started
        with _lock:
            histogram = _stage_histograms[stage]
            histogram["count"] += 1
            histogram["sum"] += seconds
            for index, boundary in enumerate(_buckets):
                histogram["buckets"][index] += int(seconds <= boundary)


def timed_stage(stage, operation, *args, **kwargs):
    """Preserve the operation's return value and exception unchanged."""
    with stage_duration(stage):
        result = operation(*args, **kwargs)
    if stage == "validate":
        citation_validation(result.validation.status)
    return result


async def measure_request(request, call_next):
    """Track completion and failures even when request handling raises an exception."""
    if request.url.path in {"/api/v1/metrics", "/metrics"}:
        return await call_next(request)
    global _resolution_active, _duration_sum
    expensive = request.method == "POST" and request.url.path.startswith(
        (
            "/api/v1/resolve",
            "/api/v1/conversation",
            "/api/v1/analyze",
            "/api/v1/retrieve",
        )
    )
    admitted = False
    start, status = perf_counter(), 500
    with _lock:
        _counts["in_flight"] += 1
        if expensive and _resolution_active < get_settings().max_resolution_requests:
            _resolution_active += 1
            admitted = True
    try:
        if expensive and hasattr(request, "client"):
            # Never trust spoofable forwarding headers without a configured trusted proxy.
            identity = sha256(
                (request.client.host if request.client else "unknown").encode()
            ).hexdigest()
            try:
                retry = await run_in_threadpool(api_budget, identity)
            except (psycopg.Error, OSError):
                status = 503
                return JSONResponse(
                    status_code=status,
                    content={"detail": "Request admission unavailable; retry shortly."},
                    headers={"Retry-After": "2"},
                )
            if retry:
                status = 429
                with _lock:
                    _counts["throttled"] += 1
                return JSONResponse(
                    status_code=status,
                    content={"detail": "Too many requests. Please wait before trying again."},
                    headers={"Retry-After": str(retry)},
                )
        if expensive and not admitted:
            status = 503
            with _lock:
                _counts["capacity_rejected"] += 1
            return JSONResponse(
                status_code=status,
                content={"detail": "Resolution capacity is busy; retry shortly."},
                headers={"Retry-After": "2"},
            )
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        with _lock:
            if admitted:
                _resolution_active -= 1
            _counts["requests"] += 1
            _counts["in_flight"] -= 1
            _counts["server_errors"] += int(status >= 500)
            _counts["client_errors"] += int(400 <= status < 500)
            seconds = perf_counter() - start
            _durations.append(seconds * 1000)
            _duration_sum += seconds
            for index, boundary in enumerate(_buckets):
                _histogram[index] += int(seconds <= boundary)


def prometheus_metrics():
    """Export bounded, content-free counters for external durable scraping."""
    with _lock:
        lines = [
            "# TYPE support_requests_total counter",
            f"support_requests_total {_counts['requests']}",
            "# TYPE support_server_errors_total counter",
            f"support_server_errors_total {_counts['server_errors']}",
            "# TYPE support_throttled_total counter",
            f"support_throttled_total {_counts['throttled']}",
            "# TYPE support_capacity_rejected_total counter",
            f"support_capacity_rejected_total {_counts['capacity_rejected']}",
            "# TYPE support_in_flight gauge",
            f"support_in_flight {_counts['in_flight']}",
            "# TYPE support_request_duration_seconds histogram",
        ]
        lines += [
            f'support_request_duration_seconds_bucket{{le="{bound}"}} {count}'
            for bound, count in zip(_buckets, _histogram)
        ]
        lines += [
            f'support_request_duration_seconds_bucket{{le="+Inf"}} {_counts["requests"]}',
            f"support_request_duration_seconds_sum {_duration_sum}",
            f"support_request_duration_seconds_count {_counts['requests']}",
        ]
        lines += ["# TYPE resolve_provider_total counter"]
        lines += [
            f'resolve_provider_total{{provider="{provider}",outcome="{outcome}"}} {count}'
            for (provider, outcome), count in sorted(_provider_counts.items())
        ]
        lines += [
            "# TYPE resolve_fallback_total counter",
            f"resolve_fallback_total {_fallback_count}",
            "# TYPE citation_validation_total counter",
        ]
        lines += [
            f'citation_validation_total{{status="{status}"}} {count}'
            for status, count in sorted(_citation_counts.items())
        ]
        lines += ["# TYPE solution_cache_total counter"]
        lines += [
            f'solution_cache_total{{outcome="{outcome}"}} {_cache_counts[outcome]}'
            for outcome in ("hit", "miss")
        ]
        lines += ["# TYPE stage_duration_seconds histogram"]
        for stage, histogram in _stage_histograms.items():
            lines += [
                f'stage_duration_seconds_bucket{{stage="{stage}",le="{boundary}"}} {count}'
                for boundary, count in zip(_buckets, histogram["buckets"], strict=True)
            ]
            lines += [
                f'stage_duration_seconds_bucket{{stage="{stage}",le="+Inf"}} {histogram["count"]}',
                f'stage_duration_seconds_sum{{stage="{stage}"}} {histogram["sum"]}',
                f'stage_duration_seconds_count{{stage="{stage}"}} {histogram["count"]}',
            ]
    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


@router.get("/metrics")
def metrics():
    """Expose aggregate latency; these counters reset when this process restarts."""
    with _lock:
        durations = sorted(_durations)
        return {
            **_counts,
            "latency_samples": len(durations),
            "p95_ms": durations[min(len(durations) - 1, int(len(durations) * 0.95))]
            if durations
            else None,
            "scope": "process_local_last_1000_requests",
            "language": language_metrics(),
            "database_pool": pool_stats(),
            "admission_backend": get_settings().rate_limit_backend,
        }
