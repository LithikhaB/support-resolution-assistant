"""Fail over from Groq to Gemini with bounded caching and short failure circuits."""

import json
from collections import OrderedDict
from contextlib import nullcontext
from contextvars import ContextVar
from functools import lru_cache
from hashlib import sha256
from threading import Lock
from time import monotonic

import psycopg

from app.config.settings import get_settings
from app.llm.base import JSONProvider
from app.llm.client import GroqClient, LanguageUnavailable
from app.llm.contracts import LanguageProvider
from app.llm.disk_cache import DiskCache
from app.llm.gemini import GeminiClient
from app.llm.privacy import redact, restore
from app.llm.telemetry import event
from app.monitoring.budgets import reserve_generation

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
            else (
                [GroqClient(self.settings)]
                if self.settings.llm_split_review
                else [GroqClient(self.settings), GeminiClient(self.settings)]
            )
        )
        self.clock = clock
        self.lock = Lock()
        self.circuits = {}
        self.rate_failures = {}
        self.cache = OrderedDict()
        self.in_flight = set()
        self.disk = DiskCache(
            self.settings.data_dir / "cache" / "wording",
            self.settings.solution_cache_seconds if self.settings.solution_cache_enabled else 0,
        )

    def generate(self, instruction, payload, schema, *, validator=None):
        # Collapse concurrent identical generations without a waiting queue.
        # The caller immediately receives its evidence-controlled local fallback.
        masked, _ = redact(payload)
        key = sha256(
            json.dumps([instruction, masked, schema.model_json_schema()], sort_keys=True).encode()
        ).hexdigest()
        with self.lock:
            if key in self.in_flight:
                raise LanguageUnavailable("duplicate_generation_in_flight")
            self.in_flight.add(key)
        try:
            return self._generate(instruction, payload, schema, validator=validator)
        finally:
            with self.lock:
                self.in_flight.discard(key)

    def _generate(self, instruction, payload, schema, *, validator=None):
        """Try each provider once and reject schema or application-level failures equally."""
        last_provider.set(None)
        masked, mapping = redact(payload)
        key = sha256(
            json.dumps([instruction, masked, schema.model_json_schema()], sort_keys=True).encode()
        ).hexdigest()
        now = self.clock()
        disk_key = sha256(
            json.dumps(
                [
                    key,
                    [(p.name, str(p.model)) for p in self.providers],
                    self.settings.llm_split_review,
                    "wording-v2",
                ]
            ).encode()
        ).hexdigest()
        with self.lock:
            cached = self.cache.get(key)
            if cached and cached[0] > now:
                self.cache.move_to_end(key)
            elif cached:
                del self.cache[key]
                cached = None
        if not cached and self.settings.solution_cache_enabled:
            stored = self.disk.get(disk_key)
            if stored:
                cached = (now + self.settings.llm_cache_seconds, stored[0], stored[1])
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
                reservation = nullcontext()
                if (
                    isinstance(provider, JSONProvider)
                    and getattr(provider.settings, f"{provider.name}_api_key").get_secret_value()
                ):
                    reservation = reserve_generation(
                        provider, getattr(validator, "reservation_calls", 2) if validator else 1
                    )
                try:
                    with reservation:
                        result = provider.generate(instruction, payload, schema)
                        if validator:
                            validator(result, provider)
                except (psycopg.Error, OSError):
                    raise LanguageUnavailable("provider_budget_unavailable") from None
                trace = {
                    "provider": provider.name,
                    "model": provider.model,
                    "cached": False,
                    "fallback": index > 0,
                }
                with self.lock:
                    self.circuits.pop(provider.name, None)
                    self.rate_failures.pop(provider.name, None)
                    if self.settings.llm_cache_seconds:
                        protected = result.model_dump(mode="json")
                        for placeholder, original in mapping.items():
                            protected = self._replace(protected, original, placeholder)
                        self.cache[key] = (
                            self.clock() + self.settings.llm_cache_seconds,
                            protected,
                            trace,
                        )
                        self.disk.put(disk_key, [protected, trace])
                        while len(self.cache) > 128:
                            self.cache.popitem(last=False)
                last_provider.set(trace)
                if index:
                    event("provider_fallbacks")
                return result
            except LanguageUnavailable as exc:
                failures.append(provider.name + ":" + str(exc))
                if str(exc).startswith("provider_http_") or str(exc) in {
                    "provider_or_schema_error",
                    "provider_throttled",
                }:
                    with self.lock:
                        delay = self.settings.llm_circuit_seconds
                        if str(exc) == "provider_throttled":
                            delay = exc.retry_after or 1
                        if str(exc) == "provider_http_429":
                            failures_count = min(6, self.rate_failures.get(provider.name, 0) + 1)
                            self.rate_failures[provider.name] = failures_count
                            delay = min(300, max(1, delay) * 2 ** (failures_count - 1))
                            delay = max(delay, exc.retry_after or 0)
                        self.circuits[provider.name] = self.clock() + delay
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
