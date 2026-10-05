"""Verify provider failover, privacy, caching and faithfulness without network access."""

from types import SimpleNamespace
from unittest.mock import Mock

from pydantic import BaseModel

from app.config.settings import Settings
from app.llm.client import LanguageUnavailable
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
    from app.monitoring.budgets import RequestBudgets

    ticks = [0]
    limiter = RequestBudgets(clock=lambda: ticks[0])
    assert limiter.take("provider", 12, 2) == 0
    assert limiter.take("provider", 12, 2) == 0
    assert limiter.take("provider", 12, 2) == 5
    ticks[0] = 5
    assert limiter.take("provider", 12, 2) == 0
    limiter.block("provider", 80)
    ticks[0] = 10
    assert limiter.take("provider", 12, 2) == 75
    ticks[0] = 85
    assert limiter.take("provider", 12, 2) == 0
    assert limiter.take("provider", 12, 2, cost=2) == 5
    ticks[0] = 90
    assert limiter.take("provider", 12, 2, cost=2) == 0
    assert limiter.take("provider", 12, 2) == 5
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
    throttled = provider("groq", error=LanguageUnavailable("provider_throttled", retry_after=15))
    chain = ProviderChain(Settings(_env_file=None), providers=[throttled], clock=lambda: 0)
    import pytest

    with pytest.raises(LanguageUnavailable, match="provider_throttled"):
        chain.generate("New", {}, Answer)
    with pytest.raises(LanguageUnavailable, match="circuit_open"):
        chain.generate("New", {}, Answer)
    assert throttled.generate.call_count == 1


def test_cached_identifier_is_restored_for_current_request_without_leaking_previous_user(tmp_path):
    from unittest.mock import patch

    from app.llm.disk_cache import DiskCache

    disk = DiskCache(tmp_path / "cache", seconds=86400, capacity=2)
    with patch("app.llm.disk_cache.time", return_value=100):
        disk.put("a", {"answer": "public evidence"})
        assert DiskCache(tmp_path / "cache").get("a") == {"answer": "public evidence"}
    with patch("app.llm.disk_cache.time", return_value=86501):
        assert disk.get("a") is None
    (tmp_path / "cache" / "bad.json").write_text("invalid json")
    assert disk.get("bad") is None
    for key in ("b", "c", "d"):
        disk.put(key, key)
    assert len(list((tmp_path / "cache").glob("*.json"))) == 2
    settings = Settings(_env_file=None, solution_cache_enabled=True, data_dir=tmp_path)
    first = provider("groq")
    ProviderChain(settings, providers=[first]).generate("persistent", {}, Answer)
    second = provider("groq", error=AssertionError("cache miss"))
    assert ProviderChain(settings, providers=[second]).generate("persistent", {}, Answer).text
    assert not second.generate.called
    split = ProviderChain(Settings(_env_file=None, llm_split_review=True))
    assert [p.name for p in split.providers] == ["groq"]

    groq = provider("groq", Answer(text="Contact first@example.com"))
    chain = ProviderChain(Settings(_env_file=None), providers=[groq])
    chain.generate("Extract", {"text": "first@example.com"}, Answer)
    result = chain.generate("Extract", {"text": "second@example.com"}, Answer)
    assert result.text == "Contact second@example.com"
    assert groq.generate.call_count == 1
    assert "first@example.com" not in str(chain.cache)


def response_fixture():
    """Expose original validation separately from untrusted generated wording."""
    return SimpleNamespace(
        draft="Original cited draft",
        validation=SimpleNamespace(status="passed"),
        analysis=SimpleNamespace(reported_facts=[], scope_status="supported"),
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
