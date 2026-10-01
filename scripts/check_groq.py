import httpx

from app.config.settings import get_settings

s = get_settings()
if not s.groq_api_key:
    raise SystemExit("GROQ_API_KEY is empty. Add it to .env")

resp = httpx.post(
    f"{s.groq_base_url}/chat/completions",
    headers={"Authorization": f"Bearer {s.groq_api_key}"},
    json={
        "model": s.groq_model,
        "messages": [{"role": "user", "content": "Reply with exactly: groq ok"}],
        "max_tokens": 10,
    },
    timeout=30,
)
print("status:", resp.status_code)
print(resp.json()["choices"][0]["message"]["content"] if resp.is_success else resp.text)