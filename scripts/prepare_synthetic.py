"""Build synthetic corpus files without changing the database."""

import argparse
import json
from pathlib import Path

from app.config.settings import get_settings
from app.ingestion.artifacts import file_sha256
from app.ingestion.synthetic import write_dataset
from scripts.enrich_procedures import publish
from scripts.publish_baseline_update import publish as publish_baselines
from scripts.publish_telecom_v3 import publish as publish_v3


def main():
    """Run the command and report its result to the terminal."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=get_settings().corpus_dir)
    args = parser.parse_args()
    if args.output.name in {"telecom_v2", "telecom_v3", "telecom_v3_1"}:
        manifest_path = args.output / "processed/manifest.json"
        if manifest_path.exists():
            report = json.loads(manifest_path.read_text(encoding="utf-8"))
            if file_sha256(args.output / "processed/documents.jsonl") != report["output_sha256"]:
                raise ValueError(
                    "Existing v2 corpus checksum mismatch; preserve it for investigation"
                )
        else:
            report = (
                publish_baselines(Path("data/synthetic/telecom_v3"), args.output)
                if args.output.name == "telecom_v3_1"
                else publish_v3(
                    Path("data/synthetic/telecom_v2"),
                    Path("data/synthetic/telecom_v3_authoring/procedures.json"),
                    args.output,
                )
                if args.output.name == "telecom_v3"
                else publish(Path("data/synthetic/telecom_v1"), args.output)
            )
    else:
        report = write_dataset(args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
