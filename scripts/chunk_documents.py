"""Create token-aware chunks without embeddings or database writes."""

import argparse
import json
from pathlib import Path

from app.config.settings import get_settings
from app.ingestion.chunking import chunk_corpus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        help="Print the first N chunks; explicit opt-in to displaying source text",
    )
    parser.add_argument("--directory", type=Path, help="Stage a separate processed corpus")
    args = parser.parse_args()
    if not 0 <= args.preview <= 10:
        parser.error("--preview must be between 0 and 10")
    report, previews = chunk_corpus(get_settings(), args.directory, args.preview)
    print(json.dumps(report, indent=2))
    for chunk in previews:
        print(json.dumps(chunk, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
