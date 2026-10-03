"""Collect bounded process-local request health without storing customer content."""

from collections import deque
from threading import Lock
from time import perf_counter

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.config.settings import get_settings
from app.llm.telemetry import snapshot as language_metrics

router = APIRouter(prefix="/api/v1")
_lock = Lock()
_durations = deque(maxlen=1000)
_counts = {"requests": 0, "server_errors": 0, "client_errors": 0, "in_flight": 0}
_resolution_active = 0


async def measure_request(request, call_next):
    """Track completion and failures even when request handling raises an exception."""
    if request.url.path == "/api/v1/metrics":
        return await call_next(request)
    global _resolution_active
    expensive = request.method == "POST" and request.url.path.startswith(
        (
            "/api/v1/resolve",
            "/api/v1/conversation",
            "/api/v1/cases",
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
            _durations.append((perf_counter() - start) * 1000)


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
