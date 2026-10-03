"""Aggregate language usage without storing prompts, keys or generated responses."""

from collections import Counter
from threading import Lock

_lock = Lock()
_counts = Counter()
_providers = {}


def record(
    provider, model, *, elapsed_ms=0, error=None, input_tokens=0, output_tokens=0, cost=None
):
    """Track bounded provider counters and explicitly configured cost estimates."""
    with _lock:
        key = f"{provider}:{model}"
        if key not in _providers and len(_providers) >= 16:
            key = "other"
        bucket = _providers.setdefault(key, Counter())
        bucket["calls"] += 1
        bucket["failures"] += int(error is not None)
        bucket["elapsed_ms"] += elapsed_ms
        bucket["input_tokens"] += input_tokens
        bucket["output_tokens"] += output_tokens
        if cost is not None:
            bucket["estimated_cost_usd"] += cost
            bucket["costed_calls"] += 1
        if error:
            bucket["rate_limits"] += int(error == "provider_http_429")


def event(name):
    """Increment a predefined operational signal."""
    if name not in {
        "cache_hits",
        "circuit_skips",
        "provider_fallbacks",
        "extractive_fallbacks",
        "faithfulness_rejections",
    }:
        raise ValueError("unknown language metric")
    with _lock:
        _counts[name] += 1


def snapshot():
    """Return process-local totals with unknown costs left explicitly unestimated."""
    with _lock:
        return {
            "events": dict(_counts),
            "providers": {key: dict(value) for key, value in _providers.items()},
            "cost_note": "Estimates require configured per-million-token rates; absent costs are unknown, not zero. Counters reset at restart.",
        }
