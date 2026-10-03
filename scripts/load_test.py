"""Run a bounded local resolution load smoke test with separate cold and warm timings."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median
from time import perf_counter
from urllib.parse import urlparse

import httpx

from app.ingestion.artifacts import write_json


def measure(base_url):
    """Send one synthetic, non-persisted complaint and retain only aggregate results."""
    started = perf_counter()
    try:
        response = httpx.post(
            base_url + "/api/v1/resolve",
            json={
                "query": "My broadband drops. Ethernet works. All wireless devices disconnect. I already restarted the router."
            },
            timeout=180,
        )
        body = response.json() if response.is_success else {}
        return {
            "status": response.status_code,
            "ms": (perf_counter() - started) * 1000,
            "language_status": body.get("language_status"),
            "provider": body.get("language_provider"),
            "citation_status": body.get("validation", {}).get("status"),
        }
    except (httpx.HTTPError, ValueError):
        return {"status": "transport_error", "ms": (perf_counter() - started) * 1000}


def main():
    """Limit traffic to a local API and never overwrite previous measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=6)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    url = urlparse(args.url)
    if (
        url.hostname not in {"127.0.0.1", "localhost"}
        or url.scheme != "http"
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        parser.error("use a plain local http URL")
    if not 1 <= args.requests <= 50 or not 1 <= args.concurrency <= 4 or args.output.exists():
        parser.error("use 1-50 requests, 1-4 workers and a new output file")
    cold = measure(args.url.rstrip("/"))
    started = perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        warm = list(pool.map(lambda _: measure(args.url.rstrip("/")), range(args.requests)))
    elapsed = perf_counter() - started
    durations = sorted(row["ms"] for row in warm)
    result = {
        "cold": cold,
        "warm": warm,
        "concurrency": args.concurrency,
        "warm_median_ms": median(durations),
        "warm_p95_ms": durations[min(len(durations) - 1, int(len(durations) * 0.95))],
        "completed_requests_per_second": args.requests / elapsed,
        "successful_requests": sum(row["status"] == 200 for row in warm),
        "scope": "Local repeated synthetic complaint; cache-assisted smoke load, not varied production traffic or a capacity guarantee.",
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
