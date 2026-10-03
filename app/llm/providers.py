"""Fail over from Groq to Gemini with bounded caching and short failure circuits."""

import json
from collections import OrderedDict
from contextvars import ContextVar
from functools import lru_cache
from hashlib import sha256
from threading import Lock
from time import monotonic

from app.config.settings import get_settings
from app.llm.client import GroqClient, LanguageUnavailable
from app.llm.contracts import LanguageProvider
from app.llm.gemini import GeminiClient
from app.llm.privacy import redact, restore
from app.llm.telemetry import event

last_provider = ContextVar("last_language_provider", default=None)


class ProviderChain:
    """Return the first valid provider response; callers own the extractive fallback."""

    def __init__(
        self, settings=None, *, providers: list[LanguageProvider] | None = None, clock=monotonic
    ):
        self.settings = settings or get_settings()
        self.providers = (
            providers
            if providers is not None
            else [GroqClient(self.settings), GeminiClient(self.settings)]
        )
        self.clock = clock
        self.lock = Lock()
        self.circuits = {}
        self.cache = OrderedDict()

    def generate(self, instruction, payload, schema, *, validator=None):
        """Try each provider once and reject schema or application-level failures equally."""
        last_provider.set(None)
        masked, mapping = redact(payload)
        key = sha256(
            json.dumps([instruction, masked, schema.model_json_schema()], sort_keys=True).encode()
        ).hexdigest()
        now = self.clock()
        with self.lock:
            cached = self.cache.get(key)
            if cached and cached[0] > now:
                self.cache.move_to_end(key)
            elif cached:
                del self.cache[key]
                cached = None
        if cached:
            result = schema.model_validate(restore(cached[1], mapping))
            provider = next(p for p in self.providers if p.name == cached[2]["provider"])
            try:
                if validator:
                    validator(result, provider)
                last_provider.set({**cached[2], "cached": True})
                event("cache_hits")
                return result
            except LanguageUnavailable:
                with self.lock:
                    self.cache.pop(key, None)
        failures = []
        for index, provider in enumerate(self.providers):
            with self.lock:
                unavailable = self.circuits.get(provider.name, 0) > self.clock()
            if unavailable:
                failures.append(provider.name + ":circuit_open")
                event("circuit_skips")
                continue
            try:
                result = provider.generate(instruction, payload, schema)
                if validator:
                    validator(result, provider)
                trace = {
                    "provider": provider.name,
                    "model": provider.model,
                    "cached": False,
                    "fallback": index > 0,
                }
                with self.lock:
                    self.circuits.pop(provider.name, None)
                    if self.settings.llm_cache_seconds:
                        protected = result.model_dump(mode="json")
                        for placeholder, original in mapping.items():
                            protected = self._replace(protected, original, placeholder)
                        self.cache[key] = (
                            self.clock() + self.settings.llm_cache_seconds,
                            protected,
                            trace,
                        )
                        while len(self.cache) > 128:
                            self.cache.popitem(last=False)
                last_provider.set(trace)
                if index:
                    event("provider_fallbacks")
                return result
            except LanguageUnavailable as exc:
                failures.append(provider.name + ":" + str(exc))
                if str(exc).startswith("provider_http_") or str(exc) == "provider_or_schema_error":
                    with self.lock:
                        self.circuits[provider.name] = (
                            self.clock() + self.settings.llm_circuit_seconds
                        )
        event("extractive_fallbacks")
        raise LanguageUnavailable(";".join(failures) or "no_provider_configured")

    @staticmethod
    def _replace(value, original, placeholder):
        """Avoid storing direct identifiers in cached schema values."""
        from app.llm.privacy import transform

        return transform(value, lambda text: text.replace(original, placeholder))


@lru_cache(maxsize=1)
def get_language_client():
    """Share bounded caches and rate-limit circuits inside one application process."""
    return ProviderChain()
