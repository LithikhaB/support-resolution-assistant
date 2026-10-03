"""Exercise remote failures, quote integrity and evidence boundaries without network calls."""

from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from app.config.settings import Settings
from app.llm.client import GroqClient, LanguageUnavailable
from app.resolution.history import select_history
from app.understanding.language import Interpretation, interpret, interpret_complaint


def test_provider_failure_never_discloses_private_body_or_key():
    client = GroqClient(
        Settings(_env_file=None, groq_api_key="secret-test"),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(429, text="private customer text secret-test")
        ),
    )
    with pytest.raises(LanguageUnavailable, match="^provider_http_429$"):
        client.generate("Extract", {"text": "private"}, Interpretation)


@pytest.mark.parametrize(
    "body",
    [
        {
            "products": [],
            "facts": [{"name": "wired_connection", "value": "working", "quote": "invented"}],
        },
        {
            "products": [],
            "facts": [
                {"name": "wired_connection", "value": "confirmed_fault", "quote": "Cannot test"}
            ],
        },
    ],
)
def test_extraction_rejects_invented_quote_or_diagnostic_value(body):
    client = Mock()
    client.generate.return_value = Interpretation.model_validate(body)
    with pytest.raises(LanguageUnavailable):
        interpret("Cannot test", client)


def test_unavailable_test_keeps_exact_quote():
    text = "I only have a phone, so I cannot use Ethernet."
    client = Mock()
    client.generate.return_value = Interpretation(
        products=[],
        facts=[
            {
                "name": "wired_connection",
                "value": "unavailable",
                "quote": text,
            }
        ],
    )
    _, facts = interpret(text, client)
    assert facts[0].value == "unavailable"
    assert text[facts[0].start : facts[0].end] == facts[0].text


def test_category_proposal_requires_local_candidate_and_affected_service_agreement():
    text = "My Wi-Fi barely works upstairs."
    client = Mock()
    client.generate.return_value = Interpretation(
        products=[{"product": "home_wifi", "quote": "Wi-Fi"}],
        facts=[],
        category={"category": "wifi_connectivity", "quote": text},
    )
    _, _, category = interpret_complaint(text, client, category_options=["wifi_connectivity"])
    assert category.category == "wifi_connectivity"
    with pytest.raises(LanguageUnavailable, match="unsupported_category_proposal"):
        interpret_complaint(text, client, category_options=["billing_dispute"])


@pytest.mark.parametrize(
    "category,quote,product",
    [
        ("billing_dispute", "My Wi-Fi fails", "home_wifi"),
        ("wifi_connectivity", "invented quote", "home_wifi"),
        ("wifi_connectivity", "My Wi-Fi fails", "landline"),
    ],
)
def test_untraceable_or_service_incompatible_category_falls_back(category, quote, product):
    client = Mock()
    client.generate.return_value = Interpretation(
        products=[{"product": product, "quote": "My Wi-Fi fails"}],
        facts=[],
        category={"category": category, "quote": quote},
    )
    with pytest.raises(LanguageUnavailable, match="unsupported_category_proposal"):
        interpret_complaint("My Wi-Fi fails", client, category_options=[category])


@pytest.mark.parametrize(
    "outcome,synthetic,refs,expected",
    [
        ("unknown", True, ["kb1"], 0),
        ("verified_resolved", True, ["kb1"], 0),
        ("simulated_resolved", True, ["other"], 0),
        ("simulated_resolved", True, ["kb1"], 1),
        ("verified_resolved", False, ["kb1"], 1),
    ],
)
def test_history_requires_resolved_outcome_and_applicable_kb(outcome, synthetic, refs, expected):
    row = SimpleNamespace(
        doc_type="resolved_ticket",
        resolution="Historical repair",
        doc_id="t1",
        chunk_id=1,
        title="Prior case",
        outcome_status=outcome,
        metadata={"is_synthetic": synthetic, "kb_refs": refs, "outcome_evidence": "Followup"},
    )
    assert len(select_history([row], [SimpleNamespace(doc_id="kb1")])) == expected
