"""Compare hybrid and reranked retrieval on development queries only."""

import argparse
import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from app.config.settings import get_settings
from app.evaluation.metrics import relevance
from app.evaluation.pipeline import verify_index
from app.ingestion.artifacts import file_sha256, write_json
from app.retrieval.models import SearchRequest
from app.retrieval.reranking import get_reranking_service
from app.retrieval.service import get_retrieval_service
from app.understanding.training import load_split


def main():
    """Use identical candidate pools, preserve per-query outcomes and leave test untouched."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("data/evaluation/reranking_development.json")
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rerank-k", type=int, default=20)
    parser.add_argument("--scope", choices=("mixed", "knowledge_base"), default="knowledge_base")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output file; existing evaluations are preserved")
    request = SearchRequest(
        query="validate bounds", rerank=True, top_k=args.top_k, rerank_k=args.rerank_k
    )
    settings = get_settings()
    snapshot = {
        "corpus_sha256": file_sha256(settings.processed_dir / "documents.jsonl"),
        "chunks_sha256": file_sha256(settings.processed_dir / "chunks.jsonl"),
    }
    verify_index(snapshot)
    rows = load_split(settings.corpus_dir, "dev")
    dev_hash = file_sha256(settings.corpus_dir / "dev.jsonl")
    service = get_retrieval_service()
    reranker = get_reranking_service()
    runs = []
    for row in rows:
        pool = service.search(
            SearchRequest(
                query=row["query"],
                top_k=request.ranking_depth,
                diversify=args.scope == "knowledge_base",
                filters={"doc_type": "knowledge_base"} if args.scope == "knowledge_base" else {},
            )
        ).results
        started = perf_counter()
        ranked = reranker.rerank(row["query"], pool, args.top_k)
        elapsed = (perf_counter() - started) * 1000
        relevant = set(row["relevant_kb_ids"])
        runs.append(
            {
                "query_id": row["query_id"],
                "scenario_family": row["scenario_family"],
                "hybrid": relevance(pool, relevant, args.top_k),
                "reranked": relevance(ranked, relevant, args.top_k),
                "candidate_hit": relevance(pool, relevant, len(pool))["hit"],
                "hybrid_doc_ids": [r.doc_id for r in pool[: args.top_k]],
                "reranked_doc_ids": [r.doc_id for r in ranked],
                "rerank_ms": elapsed,
            }
        )
        if len(runs) % 20 == 0:
            print(f"Evaluated {len(runs)}/{len(rows)} development queries", flush=True)
    summary = {
        branch: {
            "hit_at_k": mean(r[branch]["hit"] for r in runs),
            "mrr_at_k": mean(r[branch]["reciprocal_rank"] for r in runs),
        }
        for branch in ("hybrid", "reranked")
    }
    summary["candidate_hit_rate"] = mean(r["candidate_hit"] for r in runs)
    summary["median_rerank_ms"] = median(r["rerank_ms"] for r in runs)
    report = {
        "purpose": "Synthetic development comparison using provisional author-assigned KB relevance, not real-world accuracy.",
        "test_split_used": False,
        "queries": len(rows),
        "families": len({r["scenario_family"] for r in rows}),
        "dev_sha256": dev_hash,
        **snapshot,
        "scope": args.scope,
        "diversified": args.scope == "knowledge_base",
        "reranker_model": settings.reranker_model,
        "reranker_revision": settings.reranker_revision,
        "top_k": args.top_k,
        "rerank_k": args.rerank_k,
        "timing_note": "Sequential CPU calls after model initialization; includes first inference. No load-test claims.",
        "summary": summary,
        "runs": runs,
    }
    verify_index(snapshot)
    if file_sha256(settings.corpus_dir / "dev.jsonl") != dev_hash or any(
        file_sha256(settings.processed_dir / name) != snapshot[key]
        for name, key in (("documents.jsonl", "corpus_sha256"), ("chunks.jsonl", "chunks_sha256"))
    ):
        raise ValueError("Evaluation inputs changed; report was not published")
    write_json(args.output, report)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
