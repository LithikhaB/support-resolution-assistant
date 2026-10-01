from app.ingestion.cleaner import clean_text, normalize_label, normalize_sentiment, normalize_severity
from app.ingestion.schema import Sentiment, Severity


def test_clean_text_strips_html_and_whitespace():
    assert clean_text("<p>Hello   &amp;  <b>world</b>\n</p>") == "Hello & world"


def test_clean_text_handles_none():
    assert clean_text(None) == ""


def test_normalize_label():
    assert normalize_label(" Connectivity Issue ") == "connectivity_issue"
    assert normalize_label("") is None


def test_normalize_severity_aliases():
    assert normalize_severity("P1") == Severity.CRITICAL
    assert normalize_severity("HIGH") == Severity.HIGH
    assert normalize_severity("banana") is None


def test_normalize_sentiment():
    assert normalize_sentiment("Frustrated") == Sentiment.FRUSTRATED
    assert normalize_sentiment("???") is None