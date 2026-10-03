"""Measure current understanding and draft refinements on development queries only."""

import json
from collections import Counter
from statistics import mean

from app.config.settings import get_settings
from app.evaluation.preparation import fingerprint
from app.ingestion.artifacts import write_json
from app.resolution.models import ResolutionRequest
from app.resolution.service import get_resolution_service
from app.understanding.training import load_split


def main():
    """Keep the published held-out baseline intact and measure the current real service path."""
    settings = get_settings()
    before = fingerprint(settings, "dev")
    rows = load_split(settings.corpus_dir, "dev")
    service = get_resolution_service()
    runs = []
    for row in rows:
        response = service.resolve(ResolutionRequest(query=row["query"]))
        analysis = response.analysis
        runs.append(
            {
                "query_id": row["query_id"],
                "family": row["scenario_family"],
                "category": analysis.category,
                "category_basis": analysis.category_basis,
                "category_correct": analysis.category == row["labels"]["intent"],
                "severity": analysis.severity.value,
                "severity_correct": analysis.severity.value == row["labels"]["severity"],
                "has_sources": bool(response.sources),
                "draft_has_expected_kb": any(
                    s.doc_id in row["relevant_kb_ids"] for s in response.sources
                ),
                "source_ids": [s.doc_id for s in response.sources],
                "citation_validation": response.validation.status,
                "decision": response.decision.action,
            }
        )
        if len(runs) % 20 == 0:
            print(f"Evaluated {len(runs)}/{len(rows)} development queries", flush=True)
    accepted = [r for r in runs if r["category"] is not None]
    summary = {
        "accepted_categories": len(accepted),
        "total": len(runs),
        "accepted_accuracy": mean(r["category_correct"] for r in accepted) if accepted else None,
        "category_bases": dict(Counter(r["category_basis"] for r in runs)),
        "severity_accuracy": mean(r["severity_correct"] for r in runs),
        "severity_unknown_rate": mean(r["severity"] == "unknown" for r in runs),
        "drafts_with_sources": sum(r["has_sources"] for r in runs),
        "draft_expected_kb_rate": mean(r["draft_has_expected_kb"] for r in runs),
        "citation_contract_pass_rate": mean(r["citation_validation"] == "passed" for r in runs),
        "decisions": dict(Counter(r["decision"] for r in runs)),
    }
    if fingerprint(settings, "dev") != before:
        raise ValueError("Code, profile or data changed during evaluation")
    output = settings.data_dir / "evaluation/context_refinement_development_v2.json"
    write_json(
        output,
        {
            "purpose": "Development refinement results after prior held-out baseline was observed. This is not a new unbiased test evaluation.",
            "test_split_used": False,
            "fingerprint": before,
            "summary": summary,
            "runs": runs,
        },
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
