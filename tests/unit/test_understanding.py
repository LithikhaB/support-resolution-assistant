"""Challenge local understanding with negation, impact, missing models and data leakage."""

from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sklearn.linear_model import LogisticRegression

from app.api import understanding as api
from app.config.settings import Settings
from app.llm.client import LanguageUnavailable
from app.main import app
from app.understanding.classifier import (
    CategoryClassifier,
    ClassifierArtifact,
    UnderstandingUnavailable,
    make_vectorizer,
)
from app.understanding.models import AnalyzeRequest, CategoryCandidate
from app.understanding.service import UnderstandingService
from app.understanding.signals import (
    assess_sentiment,
    assess_severity,
    extract_actions,
    extract_products,
)
from app.understanding.training import classification_metrics, validate_separation


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


def test_json_classifier_matches_sklearn_without_pickles(tmp_path):
    texts = [
        "router wifi offline",
        "wireless router signal",
        "bill payment duplicate",
        "payment invoice charge",
        "mobile phone calls",
        "phone network signal",
    ]
    labels = ["wifi", "wifi", "billing", "billing", "mobile", "mobile"]
    vectorizer = make_vectorizer()
    features = vectorizer.fit_transform(texts)
    fitted = LogisticRegression().fit(features, labels)
    saved = artifact(
        classes=fitted.classes_.tolist(),
        coefficients=fitted.coef_.tolist(),
        intercept=fitted.intercept_.tolist(),
        vocabulary=vectorizer.vocabulary_,
        idf=vectorizer.idf_.tolist(),
    )
    path = tmp_path / "classifier.json"
    path.write_text(saved.model_dump_json(), encoding="utf-8")
    restored = CategoryClassifier.load(path)
    probes = ["wireless connection", "duplicate payment", "unknown vocabulary zqx"]
    assert np.allclose(
        restored.probabilities(probes), fitted.predict_proba(vectorizer.transform(probes))
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"classes": ["wifi", "wifi", "mobile"]},
        {"idf": [1, 1]},
        {"idf": [1, float("nan"), 1]},
        {"coefficients": [[1, 0], [0, 1], [1, 1]]},
        {"training_families": ["b"]},
        {"vocabulary": {"bill": 0, "mobile": 0, "wifi": 2}},
    ],
)
def test_corrupt_artifacts_rejected(changes):
    with pytest.raises(ValidationError):
        artifact(**changes)


def test_missing_and_malformed_artifacts_are_service_errors(tmp_path):
    path = tmp_path / "classifier.json"
    with pytest.raises(UnderstandingUnavailable):
        CategoryClassifier.load(path)
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(UnderstandingUnavailable):
        CategoryClassifier.load(path)


def test_embedding_revision_mismatch_prevents_inference():
    saved = artifact(
        feature_type="minilm",
        vocabulary={},
        idf=[],
        coefficients=[[0] * 384 for _ in range(3)],
        embedding_model="different",
        embedding_revision="old",
    )
    with pytest.raises(UnderstandingUnavailable):
        CategoryClassifier(saved)


@pytest.mark.parametrize(
    "text,status",
    [
        ("I already restarted the router twice; it still fails.", "attempted"),
        ("I haven't restarted the router yet.", "not_attempted"),
        ("I have not rebooted the router.", "not_attempted"),
        ("Support told me to restart the router.", "suggested"),
        ("I will reboot the router tonight.", "suggested"),
        ("Should I reboot the router?", "suggested"),
        ("If I restarted the router, would it help?", "suggested"),
    ],
)
def test_action_status_preserves_negation_and_hypotheticals(text, status):
    observations = extract_actions(text)
    assert len(observations) == 1
    assert observations[0].status == status
    assert text[observations[0].start : observations[0].end] == observations[0].text


def test_action_negation_does_not_cross_but():
    text = "I haven't checked the cable but I rebooted the router."
    actions = {item.action: item.status for item in extract_actions(text)}
    assert actions == {"check_cables": "not_attempted", "restart_device": "attempted"}


def test_tone_does_not_determine_impact():
    angry = "I'm furious about this invoice, but every service works."
    calm = "Reporting calmly: our street and neighbouring blocks all lost broadband at once."
    assert assess_sentiment(angry).value == "angry"
    assert assess_severity(angry).value == "low"
    assert assess_sentiment(calm).value == "neutral"
    assert assess_severity(calm).value == "critical"
    assert assess_severity("I am furious.").value == "unknown"
    assert assess_sentiment("I am not angry.").value == "unknown"


def test_negated_or_unrelated_outage_does_not_create_critical_impact():
    assert assess_severity("Our street has not lost service.").value != "critical"
    assert assess_severity("Our street has a shop. My phone is offline.").value != "critical"


def test_multiple_products_and_unicode_offsets():
    text = "Café: Wi-Fi fails but Ethernet works; the mobile phone is fine."
    observations = extract_products(text)
    assert {p.product for p in observations} == {"home_wifi", "broadband", "mobile"}
    assert all(text[p.start : p.end] == p.text for p in observations)


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


def test_language_assisted_category_is_distinct_from_local_confidence():
    from app.understanding.language import Interpretation

    classifier = Mock()
    classifier.artifact = artifact()
    classifier.predict.return_value = [
        CategoryCandidate(category="wifi_connectivity", score=0.2),
        CategoryCandidate(category="broadband_outage", score=0.19),
    ]
    client = Mock()
    text = "My Wi-Fi barely works upstairs."
    client.generate.return_value = Interpretation(
        products=[{"product": "home_wifi", "quote": "Wi-Fi"}],
        facts=[],
        category={"category": "wifi_connectivity", "quote": text},
    )
    service = UnderstandingService(
        classifier, settings=Settings(_env_file=None, llm_enabled=True), language=client
    )
    result = service.analyze(AnalyzeRequest(query=text))
    assert result.category == "wifi_connectivity" and result.category_basis == "language_assisted"
    assert result.candidates[0].score == 0.2
    assert result.category_evidence[0].text == text
    client.generate.side_effect = LanguageUnavailable("provider_http_429")
    fallback = service.analyze(AnalyzeRequest(query=text))
    assert fallback.category is None and fallback.language_method == "rules_fallback"


def test_empty_language_services_do_not_erase_local_billing_context():
    from app.understanding.language import Interpretation

    classifier = Mock()
    classifier.artifact = artifact()
    classifier.predict.return_value = [
        CategoryCandidate(category="billing_dispute", score=0.8),
        CategoryCandidate(category="payment_restoration", score=0.1),
    ]
    client = Mock()
    client.generate.return_value = Interpretation(products=[], facts=[])
    service = UnderstandingService(
        classifier, settings=Settings(_env_file=None, llm_enabled=True), language=client
    )
    result = service.analyze(
        AnalyzeRequest(query="I see two charges on my bill. Not sure whether one is a hold.")
    )
    assert result.category == "billing_dispute"
    assert {p.product for p in result.products} == {"billing"}
    assert any("pending or settled" in q for q in result.clarification_questions)
    assert not any("Which service" in q for q in result.clarification_questions)


@pytest.mark.parametrize("query", ["", " ", 123, "a" * 10001])
def test_api_rejects_invalid_input_before_loading_model(monkeypatch, query):
    factory = Mock()
    monkeypatch.setattr(api, "get_understanding_service", factory)
    response = TestClient(app).post("/api/v1/analyze", json={"query": query})
    assert response.status_code == 422
    factory.assert_not_called()


def test_api_missing_classifier_is_sanitized_and_health_still_works(monkeypatch, caplog):
    monkeypatch.setattr(
        api, "get_understanding_service", Mock(side_effect=UnderstandingUnavailable("private path"))
    )
    client = TestClient(app)
    response = client.post("/api/v1/analyze", json={"query": "private complaint"})
    assert response.status_code == 503
    assert "private" not in response.text + caplog.text
    assert client.get("/api/v1/health").status_code == 200


@pytest.mark.parametrize("overlap", ["family", "text", "label"])
def test_training_rejects_leakage_and_unseen_dev_classes(overlap):
    train = [{"query": "router failed", "scenario_family": "a", "labels": {"intent": "wifi"}}]
    dev = [{"query": "wireless unavailable", "scenario_family": "b", "labels": {"intent": "wifi"}}]
    if overlap == "family":
        dev[0]["scenario_family"] = "a"
    elif overlap == "text":
        dev[0]["query"] = " ROUTER FAILED "
    else:
        dev[0]["labels"]["intent"] = "unknown"
    with pytest.raises(ValueError):
        validate_separation(train, dev)


def test_family_metric_does_not_overweight_repeated_variants():
    rows = [{"scenario_family": "a", "labels": {"intent": "wifi"}} for _ in range(8)]
    rows += [{"scenario_family": "b", "labels": {"intent": "mobile"}}]
    metrics = classification_metrics(rows, ["wifi"] * 9, ["wifi", "mobile"])
    assert metrics["accuracy"] == pytest.approx(8 / 9)
    assert metrics["family_mean_accuracy"] == 0.5
