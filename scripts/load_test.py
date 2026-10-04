"""Run a bounded local resolution load smoke test with separate cold and warm timings."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from math import ceil
from pathlib import Path
from statistics import median
from time import perf_counter
from urllib.parse import urlparse

import httpx

from app.ingestion.artifacts import write_json

SMOKE_QUERY = "My broadband drops. Ethernet works. All wireless devices disconnect. I already restarted the router."


def measure(base_url, query=SMOKE_QUERY):
    """Send one synthetic, non-persisted complaint and retain only aggregate results."""
    started = perf_counter()
    try:
        response = httpx.post(
            base_url + "/api/v1/resolve",
            json={"query": query},
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
    parser.add_argument(
        "--queries", type=Path, help="JSONL complaints for varied-input measurements"
    )
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
    queries = [SMOKE_QUERY]
    if args.queries:
        queries = [
            json.loads(line)["query"]
            for line in args.queries.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not queries or any(not isinstance(query, str) or not query.strip() for query in queries):
            parser.error("queries must contain nonempty complaint strings")
    cold = measure(args.url.rstrip("/"), queries[0])
    started = perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        warm = list(
            pool.map(
                lambda index: measure(args.url.rstrip("/"), queries[index % len(queries)]),
                range(args.requests),
            )
        )
    elapsed = perf_counter() - started
    durations = sorted(row["ms"] for row in warm)
    result = {
        "cold": cold,
        "warm": warm,
        "concurrency": args.concurrency,
        "warm_median_ms": median(durations),
        "warm_p95_ms": durations[ceil(len(durations) * 0.95) - 1],
        "successful_p95_ms": (
            lambda values: values[ceil(len(values) * 0.95) - 1] if values else None
        )(sorted(row["ms"] for row in warm if row["status"] == 200)),
        "distinct_queries": len(
            set(queries[index % len(queries)] for index in range(args.requests))
        ),
        "status_counts": {
            str(status): sum(row["status"] == status for row in warm)
            for status in {row["status"] for row in warm}
        },
        "completed_requests_per_second": args.requests / elapsed,
        "successful_requests": sum(row["status"] == 200 for row in warm),
        "scope": "Local synthetic-input load measurement; inspect distinct_queries and provider/fallback fields. Not a production capacity guarantee.",
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
