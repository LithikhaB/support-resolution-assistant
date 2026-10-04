"""Prepare missing demo artifacts and start one API worker in the container."""

import subprocess
import sys

from app.config.settings import get_settings


def run(module, *args):
    subprocess.run([sys.executable, "-m", "scripts." + module, *args], check=True)


def main():
    settings = get_settings()
    run("setup_database")
    if not (settings.processed_dir / "documents.jsonl").exists():
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
