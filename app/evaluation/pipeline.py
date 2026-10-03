"""Evaluate frozen classifiers, retrieval alternatives and the local draft contract."""

import json
from collections import Counter
from datetime import datetime, timezone
from statistics import mean
from time import perf_counter

import numpy as np

from app.database.connection import get_connection
from app.evaluation.metrics import relevance, routing_metrics
from app.evaluation.preparation import (
    fingerprint,
    freeze_run,
    prepare_classifiers,
    validate_holdout,
)
from app.ingestion.artifacts import digest, file_sha256, write_json
from app.resolution.drafting import draft_resolution
from app.resolution.validation import finalize_resolution
from app.retrieval.diversity import diverse_results
from app.retrieval.models import SearchRequest
from app.retrieval.reranking import get_reranking_service
from app.retrieval.service import get_retrieval_service
from app.understanding.models import AnalyzeRequest
from app.understanding.routing import compatible_category, load_policy
from app.understanding.service import UnderstandingService
from app.understanding.signals import extract_products
from app.understanding.training import classification_metrics, load_split


def verify_index(snapshot):
    """Require the live index to match the exact local corpus and chunks being evaluated."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT status,source_hash,chunks_hash FROM retrieval_index_state WHERE singleton"
        ).fetchone()
    if not row or tuple(row) != ("ready", snapshot["corpus_sha256"], snapshot["chunks_sha256"]):
        raise ValueError("Live index differs from the frozen evaluation corpus")


def compose_case(query, understanding, retrieval):
    """Evaluate the same draft components used by the API with an explicit KB evidence pool."""
    analysis = understanding.analyze(AnalyzeRequest(query=query))
    evidence = retrieval.search(
        SearchRequest(query=query, top_k=20, diversify=True, filters={"doc_type": "knowledge_base"})
    ).results
    response = finalize_resolution(draft_resolution(analysis, evidence), evidence, analysis)
    return response, evidence


def evaluate_pipeline(settings, *, split, output, progress=print):
    """Freeze configuration, run comparisons, and publish all outcomes without retuning."""
    snapshot = fingerprint(settings, split)
    freeze_path = freeze_run(output, snapshot)
    selected, vectorizer, lexical, train, dev = prepare_classifiers(settings)
    verify_index(snapshot)
    rows = load_split(settings.corpus_dir, split)
    validate_holdout(rows, train, dev, split)
    if file_sha256(settings.corpus_dir / f"{split}.jsonl") != snapshot["declared_query_sha256"]:
        raise ValueError("Evaluation queries changed after freezing")
    texts = [r["query"] for r in rows]
    scores = {
        "published_" + selected.artifact.feature_type: (
            selected.probabilities(texts),
            selected.artifact.classes,
        ),
        "tfidf_development_selected": (
            lexical.predict_proba(vectorizer.transform(texts)),
            lexical.classes_.tolist(),
        ),
    }
    classifiers = {}
    routing = load_policy(settings, digest(selected.artifact.model_dump(mode="json"))[:16])
    for name, (values, classes) in scores.items():
        predictions = [classes[int(np.argmax(v))] for v in values]
        policy = routing if name.startswith("published_") else None
        min_score = policy.min_score if policy else settings.understanding_min_score
        min_margin = policy.min_margin if policy else settings.understanding_min_margin
        eligibility = [
            compatible_category(prediction, extract_products(row["query"]))
            for row, prediction in zip(rows, predictions, strict=True)
        ]
        classifiers[name] = {
            "routing_thresholds": {
                "min_score": min_score,
                "min_margin": min_margin,
                "compatible_service_required": True,
            },
            **classification_metrics(rows, predictions, classes),
            "routing": routing_metrics(
                rows,
                values,
                classes,
                min_score,
                min_margin,
                eligible=eligibility,
            ),
        }
    understanding = UnderstandingService(selected, settings=settings)
    retrieval = get_retrieval_service()
    reranker = get_reranking_service()
    runs = []
    for row in rows:
        started = perf_counter()
        pools = {
            mode: retrieval.search(SearchRequest(query=row["query"], mode=mode, top_k=50)).results
            for mode in ("bm25", "vector", "hybrid")
        }
        response, kb = compose_case(row["query"], understanding, retrieval)
        variants = {name: pool[:5] for name, pool in pools.items()}
        variants.update(
            {
                "diverse_hybrid": diverse_results(pools["hybrid"], 5),
                "kb_focused": kb[:5],
                "kb_reranked": reranker.rerank(row["query"], kb, 5) if kb else [],
            }
        )
        relevant = set(row["relevant_kb_ids"])
        runs.append(
            {
                "query_id": row["query_id"],
                "scenario_family": row["scenario_family"],
                "retrieval": {
                    name: relevance(results, relevant) for name, results in variants.items()
                },
                "doc_ids": {
                    name: [r.doc_id for r in results] for name, results in variants.items()
                },
                "draft_has_expected_kb": any(s.doc_id in relevant for s in response.sources),
                "citation_validation": response.validation.status,
                "cited_sources": len(response.sources),
                "decision": response.decision.model_dump(),
                "status": response.status,
                "severity_correct": response.analysis.severity.value == row["labels"]["severity"],
                "sentiment_correct": response.analysis.sentiment.value
                == row["labels"]["sentiment"],
                "severity_unknown": response.analysis.severity.value == "unknown",
                "sentiment_unknown": response.analysis.sentiment.value == "unknown",
                "comparison_ms": (perf_counter() - started) * 1000,
            }
        )
        if len(runs) % 20 == 0:
            progress(f"Evaluated {len(runs)}/{len(rows)} {split} queries", flush=True)
    challenges = []
    challenge_path = settings.corpus_dir / "challenge_queries.jsonl"
    manifest = json.loads((settings.corpus_dir / "quality_report.json").read_text(encoding="utf-8"))
    if file_sha256(challenge_path) != manifest["file_sha256"][challenge_path.name]:
        raise ValueError("Challenge checksum mismatch")
    for line in challenge_path.read_text(encoding="utf-8").splitlines():
        case = json.loads(line)
        response, _ = compose_case(case["query"], understanding, retrieval)
        challenges.append(
            {
                "query_id": case["query_id"],
                "expected_behavior": case["expected_behavior"],
                "manual_behavior_review_required": True,
                "severity_matches": response.analysis.severity.value == case["expected_severity"]
                if "expected_severity" in case
                else None,
                "sentiment_matches": response.analysis.sentiment.value == case["expected_sentiment"]
                if "expected_sentiment" in case
                else None,
                "response": response.model_dump(mode="json"),
            }
        )
    summary = {
        "retrieval": {
            name: {
                "hit_at_5": mean(r["retrieval"][name]["hit"] for r in runs),
                "mrr_at_5": mean(r["retrieval"][name]["reciprocal_rank"] for r in runs),
            }
            for name in variants
        },
        "draft_expected_kb_rate": mean(r["draft_has_expected_kb"] for r in runs),
        "citation_contract_pass_rate": mean(r["citation_validation"] == "passed" for r in runs),
        "drafts_with_sources": sum(r["cited_sources"] > 0 for r in runs),
        "decisions": dict(Counter(r["decision"]["action"] for r in runs)),
        "signals": {
            name: mean(r[name] for r in runs)
            for name in (
                "severity_correct",
                "sentiment_correct",
                "severity_unknown",
                "sentiment_unknown",
            )
        },
        "comparison_latency_ms": {
            "p50": float(np.percentile([r["comparison_ms"] for r in runs], 50)),
            "p95": float(np.percentile([r["comparison_ms"] for r in runs], 95)),
        },
    }
    verify_index(snapshot)
    if (
        fingerprint(settings, split) != snapshot
        or file_sha256(settings.corpus_dir / f"{split}.jsonl") != snapshot["declared_query_sha256"]
    ):
        raise ValueError("Inputs changed during evaluation; report was not published")
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "freeze_file": str(freeze_path),
        "queries": len(rows),
        "families": len({r["scenario_family"] for r in rows}),
        "fingerprint": snapshot,
        "classifier_comparisons": classifiers,
        "tfidf_C_selected_on_development": lexical.C,
        "summary": summary,
        "runs": runs,
        "challenge_review": challenges,
        "limitations": [
            "Synthetic authored labels and correlated variants; not real-world resolution accuracy.",
            "KB-only filtering is a narrower procedure-retrieval task than mixed retrieval.",
            "Citation contract pass rate tests extractive support, not diagnosis, relevance or arbitrary-text entailment.",
            "Latency is for the entire sequential comparison, not API request latency or a load test.",
            "Challenge behaviors require human review; source validity alone does not imply correct behavior.",
            "Test results must not be used to tune this frozen evaluation; subsequent reruns are not fresh holdouts.",
        ],
    }
    write_json(output, report)
    return report
