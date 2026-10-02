"""Atomic local artifact writes and source checksums."""

import hashlib
import json
import os
import tempfile
from pathlib import Path


def digest(value: object) -> str:
    """Hash canonical JSON for reproducible document and model fingerprints."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_write(path: Path, lines) -> None:
    """Replace a file only after all generated content has been written successfully."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
        ) as f:
            temporary = Path(f.name)
            for line in lines:
                f.write(line)
            f.flush()
            os.fsync(f.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path: Path, value: object) -> None:
    """Write an indented JSON artifact using atomic replacement."""
    atomic_write(path, [json.dumps(value, indent=2, ensure_ascii=False) + "\n"])
