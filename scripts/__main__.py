"""One entry point for setup, training, evaluation and operational commands."""

import argparse
import runpy
import sys

COMMANDS = {
    "setup": "setup_database",
    "prepare": "prepare_synthetic",
    "publish-v3": "publish_telecom_v3",
    "publish-baselines": "publish_baseline_update",
    "chunk": "chunk_documents",
    "index": "index_documents",
    "check": "check_index",
    "train": "train_understanding",
    "calibrate": "calibrate_routing",
    "reranker": "prepare_reranker",
    "analyze": "analyze_complaint",
    "resolve": "resolve_complaint",
    "converse": "converse",
    "evaluate": "evaluate_pipeline",
    "audit": "evaluate_resolution_path",
    "quality": "evaluate_response_quality",
    "review": "review_quality_report",
    "ratings": "score_response_quality",
    "load": "load_test",
    "demo": "verify_demo",
    "stage": "stage_corpus_update",
    "evolve": "demonstrate_evolution",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    module = "scripts." + COMMANDS[args.command]
    sys.argv = [module, *args.arguments]
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
