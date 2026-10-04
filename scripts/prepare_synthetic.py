"""Build synthetic corpus files without changing the database."""

import argparse
import json
from pathlib import Path

from app.config.settings import get_settings
from app.ingestion.synthetic import write_dataset


def main():
    """Run the command and report its result to the terminal."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=get_settings().corpus_dir)
    args = parser.parse_args()
    print(json.dumps(write_dataset(args.output), indent=2))


if __name__ == "__main__":
    main()
