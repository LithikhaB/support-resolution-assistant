"""Show resolution sources before and after an authenticated live DNS article ingest."""

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

import httpx

from app.config.settings import get_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--fixture", type=Path, default=Path("data/evolution/dns_category_demo.json")
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
    key = get_settings().ingest_admin_key.get_secret_value()
    if not key:
        parser.exit(2, "Set INGEST_ADMIN_KEY in the environment or private .env first.\n")
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    try:
        with httpx.Client(base_url=args.url, timeout=180) as client:
            before = client.post("/api/v1/resolve", json={"query": fixture["query"]})
            before.raise_for_status()
            update = client.post(
                "/api/v1/ingest",
                json={"documents": [fixture["knowledge"]]},
                headers={"X-Admin-Key": key},
            )
            update.raise_for_status()
            after = client.post("/api/v1/resolve", json={"query": fixture["query"]})
            after.raise_for_status()
    except httpx.HTTPError:
        parser.exit(
            1,
            "Live ingest demo failed; check readiness, admin key and model/index configuration.\n",
        )
    print("Before sources:", [source["doc_id"] for source in before.json()["sources"]])
    print("Ingest:", json.dumps(update.json(), indent=2))
    print("After sources:", [source["doc_id"] for source in after.json()["sources"]])
    if fixture["knowledge"]["doc_id"] not in {
        source["doc_id"] for source in after.json()["sources"]
    }:
        raise SystemExit(
            "New article is searchable, but was not retained in this resolution; review source relevance."
        )


if __name__ == "__main__":
    main()
