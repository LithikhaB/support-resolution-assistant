"""Test behavioural measurement, overload handling and extensible routing contracts."""

import asyncio
from types import SimpleNamespace

from fastapi.responses import JSONResponse

from app.monitoring.metrics import measure_request


def test_overload_returns_retry_hint_and_releases_capacity(monkeypatch):
    monkeypatch.setattr(
        "app.monitoring.metrics.get_settings", lambda: SimpleNamespace(max_resolution_requests=1)
    )

    async def scenario():
        """Hold one request while checking rejection of the second."""
        entered, release = asyncio.Event(), asyncio.Event()
        request = SimpleNamespace(method="POST", url=SimpleNamespace(path="/api/v1/resolve"))

        async def slow(_):
            """Return only after the competing request is checked."""
            entered.set()
            await release.wait()
            return JSONResponse({"status": "ok"})

        active = asyncio.create_task(measure_request(request, slow))
        await entered.wait()
        try:
            busy = await measure_request(request, slow)
            assert busy.status_code == 503 and busy.headers["retry-after"] == "2"
        finally:
            release.set()
            assert (await active).status_code == 200

        async def quick(_):
            """Confirm capacity is available again after completion."""
            return JSONResponse({"status": "ok"})

        assert (await measure_request(request, quick)).status_code == 200

    asyncio.run(scenario())
    from starlette.requests import Request

    from app.monitoring import metrics

    monkeypatch.setattr(metrics, "api_budget", lambda identity: 17)
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/resolve",
            "headers": [],
            "client": ("127.0.0.1", 1234),
            "scheme": "http",
        }
    )

    async def throttled():
        async def unreachable(_):
            raise AssertionError("Throttled requests must not invoke inference")

        response = await measure_request(request, unreachable)
        assert response.status_code == 429 and response.headers["retry-after"] == "17"
        assert metrics._resolution_active == 0

    asyncio.run(throttled())
