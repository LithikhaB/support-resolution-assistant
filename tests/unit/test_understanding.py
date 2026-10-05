"""Challenge local understanding with negation, impact, missing models and data leakage."""

from unittest.mock import Mock

import numpy as np

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
