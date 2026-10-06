"""Clear reusable response caches; preserve models, indexes and conversations."""

import argparse
import json

from app.config.settings import get_settings

COMPONENTS = ("evidence", "wording", "reviews")


def clear_response_caches(settings, *, dry_run=False):
    data = settings.data_dir.resolve()
    if (data / "cache").is_symlink():
        raise ValueError("Response cache directory cannot be a symbolic link")
    root = (data / "cache").resolve()
    if not root.is_relative_to(data):
        raise ValueError("Response cache directory must remain within DATA_DIR")
    files = {}
    for name in COMPONENTS:
        if (root / name).is_symlink():
            raise ValueError("Response cache component cannot be a symbolic link")
        directory = (root / name).resolve()
        if not directory.is_relative_to(root):
            raise ValueError("Response cache component escapes the cache directory")
        paths = [p for p in directory.glob("*") if p.is_file() and p.suffix in {".json", ".tmp"}]
        if any(not p.resolve().is_relative_to(directory) for p in paths):
            raise ValueError("Response cache file escapes its component directory")
        files[name] = paths
    if not dry_run:
        for paths in files.values():
            for path in paths:
                path.unlink(missing_ok=True)
    return {
        "dry_run": dry_run,
        "files": {name: len(paths) for name, paths in files.items()},
        "restart_api_to_clear_in_memory_caches": True,
        "preserved": "Model downloads, embeddings/index, corpus, conversations and rate-limit budgets",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(clear_response_caches(get_settings(), dry_run=args.dry_run), indent=2))


if __name__ == "__main__":
    main()
