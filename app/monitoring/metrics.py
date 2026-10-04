"""Collect bounded process-local request health without storing customer content."""

from collections import deque
from threading import Lock
from time import perf_counter

from fastapi import APIRouter
from fastapi.responses import JSONResponse, PlainTextResponse

from app.config.settings import get_settings
from app.llm.telemetry import snapshot as language_metrics

router = APIRouter(prefix="/api/v1")
_lock = Lock()
_durations = deque(maxlen=1000)
_counts = {"requests": 0, "server_errors": 0, "client_errors": 0, "in_flight": 0}
_resolution_active = 0
_buckets = (0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 120)
_histogram = [0] * len(_buckets)
_duration_sum = 0.0


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
        if expensive and not admitted:
            status = 503
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
        }
