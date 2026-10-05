from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from app.config.settings import Settings
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
    tokenizer.encode.side_effect = lambda text: SimpleNamespace(
        ids=list(range(len(text.split()) + 2))
    )
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
    before = components[1].encode.call_count
    cached = encoder.embed_query(texts[0])
    assert components[1].encode.call_count == before
    cached[0] = 99
    assert encoder.embed_query(texts[0])[0] != 99
    assert encoder.query_cache.hits >= 2
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "internet": 1, "slow": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    long_encoder = EmbeddingService(components[0], model=components[1], tokenizer=tokenizer)
    long_query = "internet " * 400 + "slow " * 100
    pooled = long_encoder.embed_query(long_query)
    assert len(pooled) == 384 and np.isclose(np.linalg.norm(pooled), 1)
    parts = [s for call in components[1].encode.call_args_list[before:] for s in call.args[0]]
    assert any("slow" in s for s in parts)
    with pytest.raises(ValueError, match="exceeds"):
        long_encoder.embed_documents([long_query])
