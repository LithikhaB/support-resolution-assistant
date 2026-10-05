"""Exercise remote failures, quote integrity and evidence boundaries without network calls."""

import httpx
import pytest

from app.config.settings import Settings
from app.llm.client import GroqClient, LanguageUnavailable
from app.understanding.language import Interpretation


def test_provider_failure_never_discloses_private_body_or_key():
    client = GroqClient(
        Settings(_env_file=None, groq_api_key="secret-test"),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(429, text="private customer text secret-test")
        ),
    )
    with pytest.raises(LanguageUnavailable, match="^provider_http_429$"):
        client.generate("Extract", {"text": "private"}, Interpretation)
