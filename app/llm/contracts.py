"""Provider-independent language contracts and sanitized errors."""

from typing import Protocol


class LanguageUnavailable(RuntimeError):
    """Signal an unavailable or invalid language response without private payloads."""


class LanguageProvider(Protocol):
    """Allow providers to change without changing understanding or resolution logic."""

    name: str
    model: str

    def generate(self, instruction, payload, schema):
        """Return a validated schema or raise a sanitized availability error."""
        ...
