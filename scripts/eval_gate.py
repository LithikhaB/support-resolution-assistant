"""Run the frozen development pipeline and reject regressions against fixed thresholds."""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.config.settings import get_settings
from app.evaluation.pipeline import evaluate_pipeline

METRICS = ("category_accuracy", "expected_kb_hit_at_5", "citation_validity_rate")


def gate_metrics(report, variant):
    """Use the published classifier, declared retrieval variant and citation contract."""
    if report["split"] != "dev" or report["language_path"] != "local_fallback_only":
        raise ValueError("gate requires an offline development evaluation")
    published = [
        value
        for name, value in report["classifier_comparisons"].items()
        if name.startswith("published_")
    ]
    if len(published) != 1:
        raise ValueError("expected exactly one published classifier")
    return {
        "category_accuracy": published[0]["accuracy"],
        "expected_kb_hit_at_5": report["summary"]["retrieval"][variant]["hit_at_5"],
        "citation_validity_rate": report["summary"]["citation_contract_pass_rate"],
    }


def check_gate(values, thresholds):
    """Equality passes; missing, nonfinite or invalid rates fail closed."""
    failures = []
    for name in METRICS:
        actual, minimum = values[name], thresholds[name]
        if (
            isinstance(actual, bool)
            or isinstance(minimum, bool)
            or not isinstance(actual, (int, float))
            or not isinstance(minimum, (int, float))
            or not math.isfinite(actual)
            or not math.isfinite(minimum)
            or not 0 <= actual <= 1
            or not 0 <= minimum <= 1
        ):
            raise ValueError("gate rates must be finite numbers between zero and one")
        if actual < minimum:
            failures.append(name)
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--thresholds", type=Path, default=Path("data/evaluation/gate_thresholds.json")
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser.add_argument(
        "--output", type=Path, default=Path(f".work/eval_gate_{stamp}_{uuid4().hex[:8]}.json")
    )
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.thresholds.read_text(encoding="utf-8"))
        # Validate the thresholds before running a costly evaluation.
        check_gate(config["thresholds"], config["thresholds"])
        settings = get_settings().model_copy(
            update={
                "llm_enabled": False,
                "llm_extraction_enabled": False,
                "llm_selection_enabled": False,
                "solution_cache_enabled": False,
                "lexical_backend": "postgres",
            }
        )
        report = evaluate_pipeline(settings, split="dev", output=args.output)
        if report["fingerprint"]["dev_sha256"] != config["dev_sha256"]:
            raise ValueError("development data differs from the frozen gate baseline")
        values = gate_metrics(report, config["retrieval_variant"])
        failures = check_gate(values, config["thresholds"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Evaluation gate failed: {type(exc).__name__}.\n")
    print(
        json.dumps(
            {
                "metrics": values,
                "thresholds": config["thresholds"],
                "failed_metrics": failures,
                "report": str(args.output),
            },
            indent=2,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
