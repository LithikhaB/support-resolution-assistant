"""Reuse private source selection, never old customer facts, diagnoses or LLM wording."""

from threading import Lock
from time import monotonic

import numpy as np

from app.ingestion.artifacts import digest
from app.resolution.memory import history_context


class SemanticReuse:
    def __init__(self, capacity=128, seconds=120, clock=monotonic):
        self.entries = []
        self.capacity, self.seconds, self.clock = capacity, seconds, clock
        self.lock = Lock()

    @staticmethod
    def signature(analysis, request, revision, settings):
        context = history_context.get()
        if not context or not analysis.category or analysis.scope_status != "supported":
            return None
        return digest(
            [
                context[0],
                revision,
                analysis.category,
                sorted(p.product for p in analysis.products),
                analysis.severity.value,
                analysis.sentiment.value,
                sorted((f.name, f.value) for f in analysis.reported_facts),
                [(a.action, a.status, a.text) for a in analysis.actions],
                [r.kind for r in analysis.requests],
                request.filters.model_dump(mode="json"),
                request.max_sources,
                request.rerank,
                settings.llm_enabled,
                settings.groq_model,
                settings.gemini_model,
            ]
        )

    def get(self, signature, vector, threshold):
        if signature is None:
            return None
        with self.lock:
            self.entries = [e for e in self.entries if e[0] > self.clock()]
            candidates = [
                e
                for e in self.entries
                if e[1] == signature and float(np.dot(vector, e[2])) >= threshold
            ]
            if not candidates:
                return None
            best = max(candidates, key=lambda e: float(np.dot(vector, e[2])))
            return [r.model_copy(deep=True) for r in best[3]]

    def put(self, signature, vector, evidence):
        if signature is None or not evidence or not self.seconds:
            return
        with self.lock:
            self.entries.append(
                (
                    self.clock() + self.seconds,
                    signature,
                    list(vector),
                    [r.model_copy(deep=True) for r in evidence],
                )
            )
            self.entries = self.entries[-self.capacity :]
