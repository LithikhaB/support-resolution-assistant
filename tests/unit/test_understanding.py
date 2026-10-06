"""Challenge local understanding with negation, impact, missing models and data leakage."""

from unittest.mock import Mock

import numpy as np
import pytest

from app.config.settings import Settings
from app.understanding.classifier import (
    CategoryClassifier,
    ClassifierArtifact,
)
from app.understanding.models import AnalyzeRequest, CategoryCandidate
from app.understanding.service import UnderstandingService
from app.understanding.signals import (
    assess_sentiment,
    assess_severity,
)


def test_training_uses_only_train_and_dev_and_publishes_reloadable_model(tmp_path, monkeypatch):
    from app.understanding import training

    settings = Settings(
        _env_file=None,
        corpus_dir=tmp_path,
        data_dir=tmp_path,
        understanding_model_path=tmp_path / "classifier.json",
    )
    groups = {}
    for split in ("train", "dev"):
        groups[split] = [
            {
                "query": f"{label} {word}",
                "query_id": f"{split}_{label}_{word}",
                "scenario_family": f"{split}_{label}",
                "labels": {"intent": label},
            }
            for label in ("billing", "mobile", "wifi")
            for word in (("broken", "failed") if split == "train" else ("unavailable",))
        ]
        (tmp_path / f"{split}.jsonl").write_text(split, encoding="utf-8")
    accessed = []

    def load(directory, split):
        accessed.append(split)
        assert split in ("train", "dev")
        return groups[split]

    monkeypatch.setattr(training, "load_split", load)
    encoder = Mock()

    def embed(texts):
        vectors = np.zeros((len(texts), 384))
        for index, text in enumerate(texts):
            vectors[index, ("billing", "mobile", "wifi").index(text.split()[0])] = 1
        return vectors.tolist()

    encoder.embed_documents.side_effect = embed
    report = training.train_classifier(settings, embedder=encoder)
    assert accessed == ["train", "dev"]
    assert not report["test_split_used"] and len(report["candidates"]) == 6
    loaded = CategoryClassifier.load(settings.understanding_model_path, settings=settings)
    loaded.embedder = encoder
    assert loaded.predict("wifi unavailable")[0].category == "wifi"


def artifact(**changes):
    return ClassifierArtifact(
        **(
            dict(
                feature_type="tfidf",
                classes=["billing", "mobile", "wifi"],
                coefficients=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                intercept=[0, 0, 0],
                vocabulary={"bill": 0, "mobile": 1, "wifi": 2},
                idf=[1, 1, 1],
                train_sha256="train",
                dev_sha256="dev",
                training_families=["a"],
                development_families=["b"],
                sklearn_version="test",
                regularization_c=1,
            )
            | changes
        )
    )


def test_tone_does_not_determine_impact():
    from app.understanding.context import extract_facts, extract_requests

    duplicate = "I paid the same broadband invoice twice. Both payments show settled."
    facts = {(f.name, f.value) for f in extract_facts(duplicate)}
    assert {("payment_scope", "duplicate_same_invoice"), ("billing_status", "settled")} <= facts
    quoted = "The lights are green when everything works. I called the helpline twice."
    assert not any(f.name == "service_recovery" for f in extract_facts(quoted))
    assert not extract_requests(quoted)
    wired = "I plugged my laptop into the router with the yellow cable, and it lost the connection."
    fact = next(f for f in extract_facts(wired) if f.name == "wired_connection")
    assert fact.value == "failing" and wired[fact.start : fact.end] == fact.text
    angry = "I'm furious about this invoice, but every service works."
    calm = "Reporting calmly: our street and neighbouring blocks all lost broadband at once."
    assert assess_sentiment(angry).value == "angry"
    assert assess_severity(angry).value == "low"
    assert assess_sentiment(calm).value == "neutral"
    assert assess_severity(calm).value == "critical"
    assert assess_severity("I am furious.").value == "unknown"
    assert assess_sentiment("I am not angry.").value == "unknown"
    for text in ("The optical box shows a red LOS light.", "LOS light is red."):
        severity = assess_severity(text)
        assert severity.value == "high"
        assert all(text[q.start : q.end] == q.text for q in severity.evidence)
    assert assess_severity("Is the LOS light red?").value == "unknown"
    assert (
        assess_severity("My medical alarm uses the landline and the landline is dead.").value
        == "critical"
    )
    assert assess_severity("Would a medical alarm fail if the line is down?").value == "unknown"
    from app.understanding.signals import extract_actions

    assert (
        extract_actions("I replaced the ONT power adapter yesterday.")[0].action
        == "replace_power_adapter"
    )


def test_uncertain_classifier_requests_clarification_without_logging_text(caplog):
    classifier = Mock()
    classifier.artifact = artifact()
    classifier.predict.return_value = [
        CategoryCandidate(category="wifi", score=0.38),
        CategoryCandidate(category="mobile", score=0.37),
    ]
    service = UnderstandingService(classifier, settings=Settings(_env_file=None))
    with caplog.at_level("INFO"):
        result = service.analyze(AnalyzeRequest(query="private complaint"))
    assert result.category is None and result.needs_clarification
    assert "private complaint" not in caplog.text
    assert result.severity.value == "unknown"
    language = Mock()
    classifier.predict.reset_mock()
    guarded = UnderstandingService(
        classifier, settings=Settings(_env_file=None, llm_enabled=True), language=language
    )
    for query in (
        "who is the prime minister of India?",
        "Paste any 400-word single-topic broadband complaint.",
        "What is the capital of France?",
        "Write a telecom complaint for me.",
        "What is photosynthesis?",
        "Explain the French revolution.",
        "How do I bake bread?",
    ):
        result = guarded.analyze(AnalyzeRequest(query=query))
        assert result.scope_status == "unsupported"
        assert result.category is None and not result.candidates
        assert not result.clarification_questions
    language.generate.assert_not_called()
    classifier.predict.assert_not_called()
    local = UnderstandingService(classifier, settings=Settings(_env_file=None))
    for text, category in (
        ("Web keeps vanishing around dinner time my wired PC stays online.", "wifi_connectivity"),
        ("Optical box shows a red LOS light.", "broadband_outage"),
        (
            "Oh brilliant, another lovely evening of no internet. Fix it by tomorrow or I'm cancelling my connection.",
            "broadband_outage",
        ),
        (
            "Internet romba slow ah irukku since morning, router restart pannitten no use.",
            "slow_broadband",
        ),
    ):
        result = local.analyze(AnalyzeRequest(query=text))
        assert result.category == category
        assert any(p.product == "broadband" for p in result.products)
        if "lovely" in text or "pannitten" in text:
            assert result.sentiment.value == "frustrated"
        if "pannitten" in text:
            assert result.actions[0].status == "attempted"
        for quote in [*result.actions, *result.sentiment.evidence]:
            assert text[quote.start : quote.end] == quote.text


def test_category_defaults_are_estimates_without_fabricated_quotes():
    from app.understanding.service import category_severity

    for category, expected in (
        ("broadband_outage", "high"),
        ("router_ont_hardware", "high"),
        ("payment_restoration", "high"),
        ("voice_call_failure", "high"),
        ("billing_dispute", "low"),
        ("sms_otp", "low"),
        ("mobile_data", "medium"),
        ("sim_esim_activation", "medium"),
        ("wifi_connectivity", "medium"),
        ("intermittent_broadband", "medium"),
        ("broadband_dns", "medium"),
    ):
        result = category_severity(category, [])
        assert result.value == expected and result.rule == "category_default"
        assert result.evidence == []
    assert category_severity(None, []).value == "unknown"
    from app.understanding.context import extract_facts

    charges = "Both payments show settled."
    assert category_severity("billing_dispute", extract_facts(charges)).value == "medium"


def test_category_default_uses_real_fact_spans_for_scope_and_impact():
    from app.understanding.context import extract_facts
    from app.understanding.models import ReportedFact
    from app.understanding.service import category_severity

    for text, category, expected in (
        ("My broadband keeps dropping. Ethernet also drops.", "intermittent_broadband", "high"),
        ("Everything is working again.", "broadband_outage", "low"),
        ("Only one wireless device is affected.", "wifi_connectivity", "low"),
        ("The smoking router needs support.", "router_ont_hardware", "critical"),
        ("My broadband is very slow.", "slow_broadband", "medium"),
    ):
        result = category_severity(category, extract_facts(text))
        assert result.value == expected and result.evidence
        assert all(text[q.start : q.end] == q.text for q in result.evidence)
    text = "Orders cannot be processed."
    fact = ReportedFact(name="business_impact", value="reported", text=text, start=0, end=len(text))
    result = category_severity("intermittent_broadband", [fact])
    assert result.value == "high" and result.evidence[0].text == text
    assert (
        category_severity("billing_dispute", extract_facts("Everything is working again.")).value
        == "low"
    )


def test_final_severity_fallback_preserves_existing_rules_and_abstention(monkeypatch):
    classifier = Mock()
    classifier.artifact = artifact()
    classifier.predict.return_value = [
        CategoryCandidate(category="router_ont_hardware", score=0.9),
        CategoryCandidate(category="slow_broadband", score=0.05),
    ]
    service = UnderstandingService(classifier, settings=Settings(_env_file=None, llm_enabled=False))
    result = service.analyze(AnalyzeRequest(query="My router needs investigation."))
    assert result.category == "router_ont_hardware"
    assert result.severity.value == "high" and result.severity.rule == "category_default"
    assert result.severity.evidence == []
    text = "My broadband drops every evening around 8 and I've already restarted the router twice, I work from home and this is costing me."
    result = service.analyze(AnalyzeRequest(query=text))
    assert result.severity.value == "high" and result.severity.rule == "reported_business_impact"
    assert all(text[q.start : q.end] == q.text for q in result.severity.evidence)
    from app.understanding.models import ReportedFact, RuleAssessment

    text = "Service is completely lost."
    fact = ReportedFact(
        name="impact",
        value="complete_loss",
        text=text,
        start=len("My router: "),
        end=len("My router: ") + len(text),
    )
    monkeypatch.setattr("app.understanding.service.extract_facts", lambda query: [fact])
    monkeypatch.setattr(
        "app.understanding.service.assess_severity",
        lambda query: RuleAssessment(value="unknown", rule="insufficient_impact_evidence"),
    )
    result = service.analyze(AnalyzeRequest(query="My router: " + text))
    assert result.severity.rule == "quoted_service_impact" and result.severity.value == "high"


@pytest.mark.parametrize(
    "query,expected",
    [
        ("Optical box shows a red LOS light.", "neutral"),
        ("My broadband drops every evening.", "neutral"),
        ("I am not angry. My broadband is slow.", "neutral"),
        ("My broadband is slow and costing me orders.", "frustrated"),
        ("I am furious about this invoice.", "angry"),
        ("I am worried about my broadband.", "concerned"),
    ],
)
def test_sentiment_defaults_to_neutral_without_inventing_evidence(query, expected):
    classifier = Mock()
    classifier.artifact = artifact()
    classifier.predict.return_value = [
        CategoryCandidate(category="wifi", score=0.38),
        CategoryCandidate(category="mobile", score=0.37),
    ]
    service = UnderstandingService(classifier, settings=Settings(_env_file=None, llm_enabled=False))
    result = service.analyze(AnalyzeRequest(query=query))
    assert result.sentiment.value == expected
    if expected == "neutral":
        assert result.sentiment.rule == "neutral_default"
        assert result.sentiment.evidence == []
    else:
        assert result.sentiment.evidence
        assert all(
            query[quote.start : quote.end] == quote.text for quote in result.sentiment.evidence
        )


def test_validated_language_sentiment_is_preserved_before_neutral_default(monkeypatch):
    from app.understanding.models import RuleAssessment, TextEvidence

    query = "My broadband drops. This is ruining my day."
    quote = "ruining my day"
    start = query.index(quote)
    assessment = RuleAssessment(
        value="frustrated",
        rule="quoted_language",
        evidence=[TextEvidence(text=quote, start=start, end=start + len(quote))],
    )
    monkeypatch.setattr(
        "app.understanding.service.interpret_complaint",
        lambda *args, **kwargs: ([], [], None, None, assessment),
    )
    classifier = Mock()
    classifier.artifact = artifact()
    classifier.predict.return_value = [
        CategoryCandidate(category="wifi", score=0.38),
        CategoryCandidate(category="mobile", score=0.37),
    ]
    service = UnderstandingService(
        classifier,
        language=Mock(),
        settings=Settings(_env_file=None, llm_enabled=True, llm_extraction_enabled=True),
    )
    result = service.analyze(AnalyzeRequest(query=query))
    assert result.sentiment == assessment
