"""Prepare missing demo artifacts and start one API worker in the container."""

import shutil
import subprocess
import sys
from pathlib import Path

from app.config.settings import get_settings


def run(module, *args):
    subprocess.run([sys.executable, "-m", "scripts." + module, *args], check=True)


def install_missing_corpus(target, bundled_root=Path("/service/bundled/synthetic")):
    """Seed a missing version through a mounted data volume without replacing history."""
    bundled = bundled_root / target.name
    if not target.exists() and bundled.is_dir():
        shutil.copytree(bundled, target)


def main():
    settings = get_settings()
    # The persistent /service/data volume masks image files after an upgrade.
    # Install only missing versions; never overwrite an existing frozen corpus.
    install_missing_corpus(settings.corpus_dir)
    for group in ("evaluation", "evolution"):
        bundled = Path("/service/bundled") / group
        if bundled.is_dir():
            target = settings.data_dir / group
            target.mkdir(parents=True, exist_ok=True)
            for source in bundled.glob("*.json"):
                if not (target / source.name).exists():
                    shutil.copy2(source, target / source.name)
    run("setup_database")
    if not (settings.processed_dir / "documents.jsonl").exists():
        if settings.corpus_dir.name == "telecom_v3_1":
            run("publish_baseline_update", "--output", str(settings.corpus_dir))
        elif settings.corpus_dir.name == "telecom_v3":
            run("publish_telecom_v3", "--output", str(settings.corpus_dir))
        elif settings.corpus_dir.name == "telecom_v2":
            run("enrich_procedures", "--output", str(settings.corpus_dir))
        else:
            run("prepare_synthetic")
    if not (settings.processed_dir / "chunks.jsonl").exists():
        run("chunk_documents")
    # Container packages can differ from the host that created the persisted index.
    # Re-embed changed configurations under the indexing lock; keep all source rows.
    run("index_documents", "--rebuild")
    if not settings.understanding_model_path.exists():
        run("train_understanding")
    if not settings.understanding_routing_path.exists():
        run("calibrate_routing")
    run("prepare_reranker")
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, workers=1)


if __name__ == "__main__":
    main()
