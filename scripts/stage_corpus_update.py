"""Prepare validated new tickets and articles without overwriting the active corpus."""

import argparse
import json
from pathlib import Path

from app.config.settings import get_settings
from app.ingestion.updates import stage_update


def main():
    """Stage additions and explicit ID-based replacements for review before indexing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("incoming", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = stage_update(
            get_settings().processed_dir / "documents.jsonl", args.incoming, args.output
        )
    except (ValueError, OSError):
        parser.exit(
            2,
            "Update rejected: check schema, unique IDs, KB references, evaluation split and a new output directory.\n",
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
