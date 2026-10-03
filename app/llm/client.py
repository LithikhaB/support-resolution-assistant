"""Adapt Groq's JSON completion API to the shared provider contract."""

import json

from app.llm.base import JSONProvider
from app.llm.contracts import LanguageProvider as LanguageProvider
from app.llm.contracts import LanguageUnavailable as LanguageUnavailable


class GroqClient(JSONProvider):
    """Submit one bounded Groq request; the chain handles failover immediately."""

    name = "groq"

    def request(self, client, key, instruction, payload):
        """Keep credentials in headers and untrusted data in a separate message."""
        body = {
            "model": self.model,
            "temperature": 0,
            "max_completion_tokens": 2500,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        }
        if self.model.startswith("openai/gpt-oss-"):
            body["reasoning_effort"] = "low"
        return client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json=body,
        )

    def parse(self, body):
        """Reject incomplete output and preserve vendor-reported token usage."""
        choice = body["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise LanguageUnavailable("incomplete_response")
        usage = body.get("usage", {})
        return (
            choice["message"]["content"],
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )
