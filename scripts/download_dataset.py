"""Download an immutable Hugging Face revision and record its provenance."""
import argparse
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi

from app.config.settings import get_settings
from app.ingestion.artifacts import file_sha256, write_json
from app.ingestion.loader import SOURCE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="main", help="Commit SHA or ref; resolved to a commit before downloading")
    args = parser.parse_args()
    revision = HfApi().dataset_info(SOURCE, revision=args.revision).sha
    if not revision:
        raise RuntimeError("Could not resolve an immutable dataset revision")
    dataset = load_dataset(SOURCE, revision=revision, split="train")
    english = dataset.filter(lambda row: (row.get("language") or "").strip().lower() == "en")
    if not len(english):
        raise ValueError("No English rows; refusing to replace the raw dataset")
    raw_dir = get_settings().raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / "customer_support_tickets.csv"
    fd, name = tempfile.mkstemp(dir=raw_dir, suffix=".csv")
    os.close(fd)
    temporary = Path(name)
    try:
        english.to_csv(str(temporary), index=False)
        checksum = file_sha256(temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    write_json(target.with_suffix(".manifest.json"), {
        "source": SOURCE, "requested_revision": args.revision, "revision": revision,
        "provenance_status": "pinned_download", "split": "train", "language": "en",
        "rows": len(english), "sha256": checksum,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    })
    print(f"Saved {len(english)} English rows at revision {revision} to {target}")


if __name__ == "__main__":
    main()
