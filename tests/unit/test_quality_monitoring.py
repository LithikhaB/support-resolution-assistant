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


def test_middleware_queues_burst_and_checks_budget_only_after_admission(monkeypatch):
    from app.monitoring import metrics

    settings = SimpleNamespace(
        max_resolution_requests=2, resolution_queue_size=20, resolution_queue_timeout_seconds=1
    )
    monkeypatch.setattr(metrics, "get_settings", lambda: settings)
    budgets = []
    monkeypatch.setattr(metrics, "api_budget", lambda identity: budgets.append(identity) or 0)

    async def scenario():
        request = SimpleNamespace(
            method="POST",
            url=SimpleNamespace(path="/api/v1/resolve"),
            client=SimpleNamespace(host="127.0.0.1"),
        )
        active, peak = 0, 0

        async def operation(_):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0.005)
                return JSONResponse({"status": "ok"})
            finally:
                active -= 1

        results = await asyncio.gather(*(measure_request(request, operation) for _ in range(20)))
        assert all(result.status_code == 200 for result in results)
        assert peak == 2 and metrics._resolution_active == 0 and len(budgets) == 20

    asyncio.run(scenario())
