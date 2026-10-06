"""Bounded process-local caches; hash keys, clone values and never cache failures."""

from collections import OrderedDict
from copy import deepcopy
from threading import RLock
from time import monotonic


class TTLCache:
    def __init__(self, capacity=256, seconds=120, clock=monotonic):
        self.capacity, self.seconds, self.clock = capacity, seconds, clock
        self.items = OrderedDict()
        self.lock = RLock()
        self.hits = self.misses = 0

    def get(self, key):
        with self.lock:
            item = self.items.get(key)
            if item and item[0] > self.clock():
                self.items.move_to_end(key)
                self.hits += 1
                return deepcopy(item[1])
            self.items.pop(key, None)
            self.misses += 1
            return None

    def put(self, key, value):
        if not self.seconds:
            return
        with self.lock:
            self.items[key] = (self.clock() + self.seconds, deepcopy(value))
            self.items.move_to_end(key)
            while len(self.items) > self.capacity:
                self.items.popitem(last=False)


def published_revision():
    """Check index readiness even on a cache hit; never serve an old active corpus."""
    import json

    from app.config.settings import get_settings
    from app.database.connection import get_connection
    from app.retrieval.vector_search import RetrievalUnavailable

    directory = get_settings().processed_dir
    with get_connection() as conn:
        if not conn.execute("SELECT pg_try_advisory_xact_lock_shared(8041, 2)").fetchone()[0]:
            raise RetrievalUnavailable("index_update_in_progress")
        # Read files only while holding the shared lock: publishers replace them under
        # the exclusive lock before committing the corresponding index revision.
        expected = (
            json.loads((directory / "manifest.json").read_text(encoding="utf-8"))["output_sha256"],
            json.loads((directory / "chunks.manifest.json").read_text(encoding="utf-8"))[
                "output_sha256"
            ],
        )
        row = conn.execute(
            "SELECT status,source_hash,chunks_hash,config_hash,completed_at FROM retrieval_index_state WHERE singleton"
        ).fetchone()
        if not row or row[0] != "ready" or tuple(row[1:3]) != expected:
            raise RetrievalUnavailable("index_not_ready_or_stale")
        return tuple(row[1:4]) + (str(row[4]),)
