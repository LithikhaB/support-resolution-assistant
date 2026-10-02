"""Build the staged synthetic corpus without changing the active HF corpus/database."""
import argparse
import json
from pathlib import Path

from app.ingestion.synthetic import write_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/synthetic/telecom_v1'))
    args = parser.parse_args()
    print(json.dumps(write_dataset(args.output), indent=2))


if __name__ == '__main__':
    main()
