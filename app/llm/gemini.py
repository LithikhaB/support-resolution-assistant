"""Adapt Gemini's native REST endpoint to the shared JSON-provider contract."""

import json

from app.llm.base import JSONProvider
from app.llm.contracts import LanguageUnavailable


class GeminiClient(JSONProvider):
    """Keep credentials in headers and validate generated JSON locally."""

    name = "gemini"

    def request(self, client, key, instruction, payload):
        """Separate trusted system instructions from the customer-data message."""
        return client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            headers={"x-goog-api-key": key},
            json={
                "systemInstruction": {"parts": [{"text": instruction}]},
                "contents": [
                    {"role": "user", "parts": [{"text": json.dumps(payload, ensure_ascii=False)}]}
                ],
                "generationConfig": {
                    "temperature": 0,
                    "maxOutputTokens": 4096,
                    "responseMimeType": "application/json",
                },
            },
        )

    def parse(self, body):
        """Reject safety-blocked or truncated output and omit thought parts."""
        candidate = body["candidates"][0]
        if candidate.get("finishReason") != "STOP":
            raise LanguageUnavailable("incomplete_response")
        text = "".join(
            part.get("text", "")
            for part in candidate["content"]["parts"]
            if not part.get("thought")
        )
        usage = body.get("usageMetadata", {})
        return (
            text,
            usage.get("promptTokenCount", 0),
            usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0),
        )
