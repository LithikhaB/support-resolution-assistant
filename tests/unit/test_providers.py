"""Verify provider failover, privacy, caching and faithfulness without network access."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from pydantic import BaseModel

from app.config.settings import Settings
from app.llm.client import GroqClient, LanguageUnavailable
from app.llm.gemini import GeminiClient
from app.llm.providers import ProviderChain, last_provider
from app.resolution.language import DraftIntroduction, FaithfulnessReview, add_language_draft
from app.resolution.models import CustomerPlan


class Answer(BaseModel):
    """Use a small schema to isolate provider behaviour."""

    text: str


def provider(name, result=None, error=None):
    """Construct a provider with explicit name and model metadata."""
    item = Mock(name=name)
    item.name, item.model = name, "test-model"
    item.generate.return_value = result or Answer(text="Supported answer")
    item.generate.side_effect = error
    return item


def test_rate_limit_moves_to_gemini_and_skips_groq_during_cooldown():
    groq = provider("groq", error=LanguageUnavailable("provider_http_429"))
    gemini = provider("gemini")
    now = [0]
    chain = ProviderChain(
        Settings(_env_file=None, llm_cache_seconds=0),
        providers=[groq, gemini],
        clock=lambda: now[0],
    )
    assert chain.generate("Draft", {}, Answer).text == "Supported answer"
    assert last_provider.get()["provider"] == "gemini"
    chain.generate("Draft", {"text": "next"}, Answer)
    assert groq.generate.call_count == 1 and gemini.generate.call_count == 2
    now[0] = 61
    chain.generate("Draft", {}, Answer)
    assert groq.generate.call_count == 2


def test_invalid_schema_or_application_response_tries_next_provider():
    groq, gemini = provider("groq", Answer(text="Invented")), provider("gemini")
    chain = ProviderChain(Settings(_env_file=None), providers=[groq, gemini])

    def validate(result, source):
        """Reject an invented claim independently of JSON syntax."""
        if result.text == "Invented":
            raise LanguageUnavailable("faithfulness_rejected")

    assert chain.generate("Draft", {}, Answer, validator=validate).text == "Supported answer"
    assert last_provider.get()["provider"] == "gemini"


def test_cached_identifier_is_restored_for_current_request_without_leaking_previous_user():
    groq = provider("groq", Answer(text="Contact first@example.com"))
    chain = ProviderChain(Settings(_env_file=None), providers=[groq])
    chain.generate("Extract", {"text": "first@example.com"}, Answer)
    result = chain.generate("Extract", {"text": "second@example.com"}, Answer)
    assert result.text == "Contact second@example.com"
    assert groq.generate.call_count == 1
    assert "first@example.com" not in str(chain.cache)


@pytest.mark.parametrize("provider_type", [GroqClient, GeminiClient])
def test_credentials_and_identifiers_stay_out_of_url_and_remote_payload(provider_type):
    captured = []

    def respond(request):
        """Return a provider-shaped response echoing a masked quote."""
        captured.append(request)
        content = json.dumps({"text": "Email [PRIVATE_1]"})
        if provider_type is GroqClient:
            body = {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}
        else:
            body = {
                "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": content}]}}]
            }
        return httpx.Response(200, json=body)

    settings = Settings(_env_file=None, groq_api_key="groq-secret", gemini_api_key="gemini-secret")
    result = provider_type(settings, transport=httpx.MockTransport(respond)).generate(
        "Extract", {"text": "Email user@example.com; account ID AC12345; +91 9876543210"}, Answer
    )
    assert result.text == "Email user@example.com"
    assert "secret" not in str(captured[0].url)
    assert all(
        value not in captured[0].content.decode()
        for value in ["user@example.com", "AC12345", "9876543210"]
    )


def response_fixture():
    """Expose original validation separately from untrusted generated wording."""
    return SimpleNamespace(
        draft="Original cited draft",
        validation=SimpleNamespace(status="passed"),
        analysis=SimpleNamespace(reported_facts=[]),
        acknowledged_actions=[],
        customer_plan=CustomerPlan(
            title="Clarify", summary="Need details", steps=["Answer the question."]
        ),
        clarification_questions=["Which device is affected?"],
        historical_cases=[],
        sources=[],
        suggestions=[],
        limitations=[],
        language_draft=None,
        faithfulness_status="not_run",
    )


def test_both_providers_down_preserve_extractive_draft():
    chain = ProviderChain(
        Settings(_env_file=None),
        providers=[
            provider("groq", error=LanguageUnavailable("provider_http_429")),
            provider("gemini", error=LanguageUnavailable("provider_http_503")),
        ],
    )
    result = add_language_draft(
        response_fixture(), "Complaint", Settings(_env_file=None, llm_enabled=True), chain
    )
    assert result.language_status == "fallback"
    assert result.language_draft is None and result.draft == "Original cited draft"
    assert result.validation.status == "passed"


@pytest.mark.parametrize(
    "summary,citations",
    [
        ("Use [S99]", []),
        ("Earlier case", ["T99"]),
        ("I'll check the network trace now.", []),
        ("I will arrange a replacement.", []),
        ("Earlier checks did not identify the issue.", []),
        ("No prior troubleshooting steps taken.", []),
    ],
)
def test_invented_citations_never_reach_display(summary, citations):
    bad = DraftIntroduction(summary=summary, history_citations=citations)
    chain = ProviderChain(
        Settings(_env_file=None), providers=[provider("groq", bad), provider("gemini", bad)]
    )
    result = add_language_draft(
        response_fixture(), "Complaint", Settings(_env_file=None, llm_enabled=True), chain
    )
    assert result.language_status == "fallback" and result.language_draft is None


def test_faithfulness_rejection_falls_through_to_supported_gemini_draft():
    groq, gemini = provider("groq"), provider("gemini")
    groq.generate.side_effect = [
        DraftIntroduction(summary="The fault is confirmed.", history_citations=[]),
        FaithfulnessReview(supported=False, issues=["unconfirmed cause"]),
    ]
    gemini.generate.side_effect = [
        DraftIntroduction(
            summary="More device details are needed before choosing a repair.", history_citations=[]
        ),
        FaithfulnessReview(supported=True, issues=[]),
    ]
    chain = ProviderChain(Settings(_env_file=None), providers=[groq, gemini])
    result = add_language_draft(
        response_fixture(), "Complaint", Settings(_env_file=None, llm_enabled=True), chain
    )
    assert result.language_provider == "gemini" and result.faithfulness_status == "model_checked"
    assert "confirmed" not in result.language_draft
    assert "Answer the question." in result.language_draft
