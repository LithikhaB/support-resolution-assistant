"""Bounded, atomic JSON cache of masked, validated provider responses."""

import json
import os
import tempfile
from pathlib import Path
from time import time


class DiskCache:
    def __init__(self, directory, seconds=86400, capacity=256):
        self.directory = Path(directory)
        self.seconds = seconds
        self.capacity = capacity

    def get(self, key):
        try:
            item = json.loads((self.directory / (key + ".json")).read_text("utf-8"))
            if item["expires"] > time():
                return item["value"]
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return None

    def put(self, key, value):
        if not self.seconds:
            return
        temporary = None
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.directory, suffix=".tmp", delete=False
            ) as stream:
                temporary = stream.name
                json.dump({"expires": time() + self.seconds, "value": value}, stream)
            os.replace(temporary, self.directory / (key + ".json"))
            files = sorted(self.directory.glob("*.json"), key=lambda p: p.stat().st_mtime)
            for path in files[: max(0, len(files) - self.capacity)]:
                path.unlink(missing_ok=True)
        except OSError:
            # Cache storage failure must not prevent a supported local response.
            pass
        finally:
            if temporary:
                Path(temporary).unlink(missing_ok=True)
