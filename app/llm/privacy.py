"""Mask common direct identifiers before remote language calls."""

import re

PATTERN = re.compile(
    r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"
    r"|\b(?:gsk_[A-Za-z0-9_-]{10,}|AIza[A-Za-z0-9_-]{10,})\b"
    r"|\b(?:password|passcode|OTP|verification code)\s*[:=]\s*\S+"
    r"|(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)"
    r"|\b(?:account|customer|ticket)\s*(?:id|number|no\.?|#)\s*[:=]?\s*[A-Za-z0-9-]{4,}",
    re.I,
)


def transform(value, replace):
    """Transform string values while preserving schema keys and data structure."""
    if isinstance(value, str):
        return replace(value)
    if isinstance(value, list):
        return [transform(item, replace) for item in value]
    if isinstance(value, dict):
        return {key: transform(item, replace) for key, item in value.items()}
    return value


def redact(payload):
    """Return reversible per-request placeholders; never retain identifiers in a cache."""
    mapping = {}
    encoded = str(payload)
    prefix = "PRIVATE"
    while f"[{prefix}_" in encoded:
        prefix += "X"

    def substitute(match):
        original = match.group()
        if original not in mapping:
            mapping[original] = f"[{prefix}_{len(mapping) + 1}]"
        return mapping[original]

    masked = transform(payload, lambda text: PATTERN.sub(substitute, text))
    return masked, {placeholder: original for original, placeholder in mapping.items()}


def restore(payload, mapping):
    """Restore quoted observations locally so their character offsets remain checkable."""

    def substitute(text):
        for placeholder, original in mapping.items():
            text = text.replace(placeholder, original)
        return text

    return transform(payload, substitute)
