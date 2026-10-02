import html
import re
from datetime import date

from app.ingestion.schema import Sentiment, Severity

_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")
_PLACEHOLDER = re.compile(
    r"<(name|your name|tel_num|acc_num|email|time|website_url|amount|tool_name|"
    r"tool[123]|company|company_name|link|price|case_num|contact_info|sn|"
    r"organization|ref_num|support team|url|forum_url|user|tool names)>", re.I
)
_ESCAPED_WHITESPACE = re.compile(r"\\r\\n|\\[nrt]")

_SEVERITY_ALIASES: dict[str, Severity] = {
    "low": Severity.LOW, "p4": Severity.LOW,
    "medium": Severity.MEDIUM, "med": Severity.MEDIUM, "p3": Severity.MEDIUM,
    "high": Severity.HIGH, "p2": Severity.HIGH,
    "critical": Severity.CRITICAL, "p1": Severity.CRITICAL,
}


def clean_text(raw: str | None) -> str:
    """Unescape HTML entities, strip tags, and collapse whitespace."""
    text = html.unescape(raw or "")
    text = _PLACEHOLDER.sub(lambda m: "[" + m.group(1).lower().replace(" ", "_") + "]", text)
    text = _HTML_TAG.sub(" ", text)
    text = _ESCAPED_WHITESPACE.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def normalize_label(raw: str | None) -> str | None:
    """'Connectivity Issue ' -> 'connectivity_issue'. Empty -> None."""
    cleaned = clean_text(raw).lower().replace(" ", "_")
    return cleaned or None


def normalize_severity(raw: str | None) -> Severity | None:
    return _SEVERITY_ALIASES.get(clean_text(raw).lower())


def normalize_sentiment(raw: str | None) -> Sentiment | None:
    try:
        return Sentiment(clean_text(raw).lower())
    except ValueError:
        return None


def parse_date(raw: str | None) -> date | None:
    try:
        return date.fromisoformat(clean_text(raw))
    except ValueError:
        return None
