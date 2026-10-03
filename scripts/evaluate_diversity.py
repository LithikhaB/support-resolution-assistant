"""Measure stable diversity and KB-focused retrieval on the synthetic development split."""

import json
from pathlib import Path
from statistics import mean

from app.config.settings import get_settings
from app.ingestion.artifacts import file_sha256, write_json
from app.retrieval.diversity import diverse_results
from app.retrieval.models import SearchRequest
from app.retrieval.service import get_retrieval_service
from app.understanding.training import load_split
from scripts.evaluate_reranking import relevance


def main():
    """Compare retrieval policies without tuning on or loading test queries."""
    settings = get_settings()
    service = get_retrieval_service()
    rows = load_split(settings.corpus_dir, "dev")
    runs = []
    for row in rows:
        pool = service.search(SearchRequest(query=row["query"], top_k=50)).results
        variants = {
            "hybrid": pool[:5],
            "diverse_hybrid": diverse_results(pool, 5),
            "kb_focused": service.search(
                SearchRequest(
                    query=row["query"],
                    top_k=5,
                    diversify=True,
                    filters={"doc_type": "knowledge_base"},
                )
            ).results,
        }
        runs.append(
            {
                "query_id": row["query_id"],
                "scenario_family": row["scenario_family"],
                **{
                    name: relevance(results, set(row["relevant_kb_ids"]), 5)
                    for name, results in variants.items()
                },
                "doc_ids": {
                    name: [r.doc_id for r in results] for name, results in variants.items()
                },
            }
        )
        if len(runs) % 20 == 0:
            print(f"Evaluated {len(runs)}/{len(rows)} development queries", flush=True)
    summary = {
        name: {
            "hit_at_5": mean(r[name]["hit"] for r in runs),
            "mrr_at_5": mean(r[name]["reciprocal_rank"] for r in runs),
        }
        for name in variants
    }
    report = {
        "purpose": "Development-only retrieval policy comparison with provisional authored KB labels; KB filtering changes the search task and is appropriate for procedure drafting, not a general search improvement claim.",
        "test_split_used": False,
        "queries": len(rows),
        "families": len({r["scenario_family"] for r in rows}),
        "dev_sha256": file_sha256(settings.corpus_dir / "dev.jsonl"),
        "corpus_sha256": file_sha256(settings.processed_dir / "documents.jsonl"),
        "summary": summary,
        "runs": runs,
    }
    output = Path("data/evaluation/diversity_development.json")
    write_json(output, report)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
