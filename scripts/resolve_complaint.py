"""Produce a local evidence-based resolution draft for agent review."""

import argparse

import psycopg
from pydantic import ValidationError

from app.resolution.models import ResolutionRequest
from app.resolution.service import get_resolution_service
from app.retrieval.embeddings import EmbeddingInputTooLong
from app.retrieval.vector_search import RetrievalUnavailable
from app.understanding.classifier import UnderstandingUnavailable


def main():
    """Print a readable draft, or JSON with analysis, source spans and conditional actions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--max-sources", type=int, default=3)
    ranking = parser.add_mutually_exclusive_group()
    ranking.add_argument("--rerank", dest="rerank", action="store_true")
    ranking.add_argument("--no-rerank", dest="rerank", action="store_false")
    parser.set_defaults(rerank=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        request = ResolutionRequest(
            query=args.query, max_sources=args.max_sources, rerank=args.rerank
        )
        response = get_resolution_service().resolve(request)
    except (ValidationError, EmbeddingInputTooLong):
        parser.exit(2, "Provide a bounded complaint and max-sources between 1 and 5.\n")
    except psycopg.errors.QueryCanceled:
        parser.exit(1, "Evidence retrieval timed out; retry.\n")
    except (UnderstandingUnavailable, RetrievalUnavailable, psycopg.Error, OSError):
        parser.exit(
            1,
            "Resolution unavailable; check the trained classifier, model cache and database index.\n",
        )
    if args.json:
        print(response.model_dump_json(indent=2))
        return
    print(response.draft)
    if response.sources:
        print("\nSources:")
        for source in response.sources:
            print(
                f"[{source.citation_id}] {source.doc_id} | chunk {source.chunk_id} | {source.title}"
            )
    print(
        f"\nCitation validation: {response.validation.status} | Checked sources: {response.validation.checked_sources}"
    )
    print(f"\nStatus: {response.status} | Method: {response.method} | {response.elapsed_ms:.1f} ms")


if __name__ == "__main__":
    main()
