"""Test behavioural measurement, overload handling and extensible routing contracts."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi.responses import JSONResponse

from app.evaluation.quality import check_snapshot, human_review_summary, summarize
from app.monitoring.metrics import measure_request
from app.understanding.routing import compatible_category, load_category_products


def test_report_errors_fail_golden_checks_instead_of_disappearing():
    assert check_snapshot({"error": "unavailable"}, {}) == ["request_error"]
    report = {"results": [{"id": "case", "snapshots": [{"error": "unavailable"}]}]}
    result = summarize(report, {"cases": [{"id": "case"}]})
    assert result["counts"]["request_errors"] == 1
    assert result["golden_checked"] == 1 and result["golden_passed"] == 0


def test_successful_final_reply_does_not_hide_an_earlier_request_failure():
    case = {
        "id": "case",
        "snapshots": [
            {"after_reply": 0, "error": "unavailable"},
            {"after_reply": 1, "response": {"issues": []}},
        ],
    }
    result = summarize({"results": [case]}, {"cases": [{"id": "case"}]})
    assert result["golden_passed"] == 0
    assert "request_error" in result["golden_results"][0]["failures"]


def test_empty_run_cannot_pass_and_missing_checkpoint_is_visible():
    result = summarize(
        {"results": [{"id": "case", "snapshots": []}]},
        {"cases": [{"id": "case", "checkpoints": [{"after_reply": 1}]}]},
    )
    assert result["golden_passed"] == 0
    assert set(result["golden_results"][0]["failures"]) == {
        "missing_snapshots",
        "missing_checkpoint",
    }


def test_unscored_human_reviews_do_not_become_zero_or_perfect_scores():
    result = human_review_summary([{"turns": [], "manual_scores": {}}])
    assert result["fully_reviewed_cases"] == 0
    assert all(d["mean_out_of_2"] is None for d in result["dimensions"].values())


def test_human_review_requires_conversation_rating_only_for_followups():
    scores = dict(context=2, relevance=1, grounding=2, clarity=1)
    result = human_review_summary(
        [{"turns": [], "manual_scores": scores}, {"turns": [{}], "manual_scores": scores}]
    )
    assert result["fully_reviewed_cases"] == 1
    assert result["dimensions"]["relevance"] == {"rated_cases": 2, "mean_out_of_2": 1}


@pytest.mark.parametrize("value", [True, 3, -1, 1.5, "2"])
def test_invalid_human_scores_are_not_silently_averaged(value):
    with pytest.raises(ValueError):
        human_review_summary([{"manual_scores": {"context": value}}])


def test_new_category_service_mapping_requires_no_logic_change(tmp_path):
    path = tmp_path / "categories.json"
    path.write_text(json.dumps({"new_wifi_category": ["home_wifi"]}), encoding="utf-8")
    mapping = load_category_products(path)
    assert compatible_category("new_wifi_category", [SimpleNamespace(product="home_wifi")], mapping)
    assert not compatible_category(
        "new_wifi_category", [SimpleNamespace(product="billing")], mapping
    )


@pytest.mark.parametrize("mapping", [{"new": ["invented_product"]}, {"new": "home_wifi"}, []])
def test_invalid_category_mappings_fail_closed(tmp_path, mapping):
    path = tmp_path / "categories.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")
    with pytest.raises(ValueError):
        load_category_products(path)


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
