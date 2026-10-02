from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from app.config.settings import Settings
from app.retrieval import embeddings
from app.retrieval.embeddings import EmbeddingService


@pytest.fixture
def components():
    settings = Settings(_env_file=None, embedding_batch_size=2)
    model = Mock()
    model.get_embedding_dimension.return_value = 384
    model.max_seq_length = 256
    def encode(texts, **kwargs):
        result = np.zeros((len(texts), 384), dtype=np.float32)
        for i, text in enumerate(texts):
            result[i, 0] = len(text)
            result[i, 1] = 1
        return result
    model.encode.side_effect = encode
    tokenizer = Mock()
    tokenizer.encode.side_effect = lambda text: SimpleNamespace(ids=list(range(len(text.split()) + 2)))
    return settings, model, tokenizer


def service(parts):
    settings, model, tokenizer = parts
    return EmbeddingService(settings, model=model, tokenizer=tokenizer)


def test_batches_preserve_order_normalize_and_reuse_model(components):
    encoder = service(components)
    texts = ["a", "longer text", "another example"]
    vectors = np.asarray(encoder.embed_documents(texts))
    assert vectors.shape == (3, 384)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1)
    assert vectors[0, 0] < vectors[1, 0] < vectors[2, 0]
    calls = components[1].encode.call_args_list
    assert [c.args[0] for c in calls] == [texts[:2], texts[2:]]
    assert all(c.kwargs["normalize_embeddings"] for c in calls)
    assert np.allclose(encoder.embed_query(texts[0]), vectors[0])
    assert encoder.model is components[1]


def test_empty_batch_is_valid(components):
    assert service(components).embed_documents([]) == []
    components[1].encode.assert_not_called()


@pytest.mark.parametrize("text", ["", "   ", None, 123, "word " * 255])
def test_invalid_inputs_fail_before_inference(components, text):
    with pytest.raises(ValueError):
        service(components).embed_query(text)
    components[1].encode.assert_not_called()


@pytest.mark.parametrize("bad", [np.zeros((1, 383)), np.zeros((1, 384)),
                                  np.full((1,384), np.nan), np.full((1,384), np.inf)])
def test_invalid_model_outputs_rejected(components, bad):
    components[1].encode.side_effect = None
    components[1].encode.return_value = bad
    with pytest.raises(ValueError):
        service(components).embed_query("a valid input")


def test_wrong_model_dimension_rejected(components):
    components[1].get_embedding_dimension.return_value = 768
    with pytest.raises(ValueError, match="dimension"):
        service(components)


def test_configured_dimension_must_match_database(components):
    components[0].embedding_dim = 768
    with pytest.raises(ValueError, match="384"):
        service(components)


def test_concurrent_factory_initializes_once(monkeypatch):
    embeddings._cached_service.cache_clear()
    factory = Mock(return_value=object())
    monkeypatch.setattr(embeddings, "EmbeddingService", factory)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            instances = list(pool.map(lambda _: embeddings.get_embedding_service(), range(8)))
        factory.assert_called_once()
        assert all(x is instances[0] for x in instances)
    finally:
        embeddings._cached_service.cache_clear()


def test_token_limit_includes_special_tokens(components):
    assert len(service(components).embed_query("word " * 254)) == 384


def test_default_loader_uses_cpu_pinned_revision_and_local_cache(components, monkeypatch):
    import sys
    settings, model, tokenizer = components
    constructor = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=constructor))
    loader = Mock(return_value=(tokenizer, None))
    monkeypatch.setattr(embeddings, "load_tokenizer", loader)
    EmbeddingService(settings)
    constructor.assert_called_once_with(
        settings.embedding_model, device="cpu", revision=settings.tokenizer_revision,
        cache_folder=str(settings.data_dir / "models"),
        local_files_only=settings.embedding_local_files_only, trust_remote_code=False)
    loader.assert_called_once()
