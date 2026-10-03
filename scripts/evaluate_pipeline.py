"""Run frozen pipeline and model comparisons on development or held-out test queries."""

import argparse
import json
from pathlib import Path

from app.config.settings import get_settings
from app.evaluation.pipeline import evaluate_pipeline


def main():
    """Save a versioned report; refuse to overwrite previous evaluation evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(f"data/evaluation/pipeline_{args.split}.json")
    report = evaluate_pipeline(get_settings(), split=args.split, output=output)
    print(
        json.dumps(
            {
                "split": args.split,
                "summary": report["summary"],
                "classifiers": {
                    name: {
                        "accuracy": value["accuracy"],
                        "macro_f1": value["macro_f1"],
                        "routing": value["routing"],
                    }
                    for name, value in report["classifier_comparisons"].items()
                },
            },
            indent=2,
        )
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
