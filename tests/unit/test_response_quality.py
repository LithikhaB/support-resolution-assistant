"""Keep evaluation failures visible and preserve every conversation prefix without scoring claims."""

from unittest.mock import Mock

from fastapi import HTTPException

from scripts.evaluate_response_quality import evaluate_case


def test_each_prefix_is_evaluated_and_manual_scores_remain_empty(monkeypatch):
    """A final-turn-only report must not hide a repeated or incorrect intermediate question."""
    response = Mock()
    response.model_dump.return_value = {"issues": []}
    resolve = Mock(return_value=response)
    monkeypatch.setattr("scripts.evaluate_response_quality.resolve_conversation", resolve)
    case = {
        "query": "My broadband drops.",
        "turns": [
            {"issue_id": 1, "message": "Ethernet works."},
            {"issue_id": 1, "message": "Correction: Ethernet drops."},
        ],
    }
    result = evaluate_case(case, Mock())
    assert [len(call.args[0].turns) for call in resolve.call_args_list] == [0, 1, 2]
    assert [item["after_reply"] for item in result["snapshots"]] == [0, 1, 2]
    assert all(score is None for score in result["manual_scores"].values())


def test_validation_failure_is_recorded_as_a_quality_limitation(monkeypatch):
    """A rejected natural reply remains in the report instead of aborting the test pack."""
    response = Mock()
    response.model_dump.return_value = {"issues": []}
    resolve = Mock(return_value=response)
    monkeypatch.setattr("scripts.evaluate_response_quality.resolve_conversation", resolve)
    result = evaluate_case(
        {
            "query": "My broadband drops.",
            "turns": [{"issue_id": 1, "message": "My SIM cannot make calls."}],
        },
        Mock(),
    )
    assert len(result["snapshots"]) == 2
    assert result["snapshots"][1]["status"] == 422
    assert "another service" in result["snapshots"][1]["error"]
    assert resolve.call_count == 1


def test_dependency_failure_does_not_hide_following_snapshot(monkeypatch):
    """Infrastructure failures cannot silently disappear from the evaluation denominator."""
    response = Mock()
    response.model_dump.return_value = {"issues": []}
    resolve = Mock(side_effect=[HTTPException(503, "Model unavailable"), response])
    monkeypatch.setattr("scripts.evaluate_response_quality.resolve_conversation", resolve)
    result = evaluate_case(
        {"query": "My broadband drops.", "turns": [{"issue_id": 1, "message": "Ethernet works."}]},
        Mock(),
    )
    assert result["snapshots"][0]["status"] == 503
    assert "response" in result["snapshots"][1]
