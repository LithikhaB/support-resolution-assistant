"""Reject frozen dev regressions without using test data or enabling providers."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts import eval_gate


def report():
    return {
        "split": "dev",
        "language_path": "local_fallback_only",
        "fingerprint": {"dev_sha256": "frozen-dev"},
        "classifier_comparisons": {
            "published_minilm": {"accuracy": 0.8},
            "tfidf_development_selected": {"accuracy": 0.99},
        },
        "summary": {
            "retrieval": {"kb_focused": {"hit_at_5": 0.9}},
            "citation_contract_pass_rate": 1.0,
        },
    }


def test_gate_equality_passes_and_any_regression_fails():
    values = eval_gate.gate_metrics(report(), "kb_focused")
    assert values["category_accuracy"] == 0.8
    assert not eval_gate.check_gate(values, values)
    for name in eval_gate.METRICS:
        regressed = {**values, name: values[name] - 0.021}
        assert eval_gate.check_gate(regressed, {k: v - 0.02 for k, v in values.items()}) == [name]
    for value in (float("nan"), float("inf"), -1, 2, True, "0.8"):
        with pytest.raises(ValueError):
            eval_gate.check_gate({**values, "category_accuracy": value}, values)
    for split, language in (("test", "local_fallback_only"), ("dev", "generated")):
        wrong = deepcopy(report())
        wrong.update(split=split, language_path=language)
        with pytest.raises(ValueError):
            eval_gate.gate_metrics(wrong, "kb_focused")


def test_gate_main_runs_offline_pipeline_and_returns_nonzero(monkeypatch, tmp_path):
    config = tmp_path / "thresholds.json"
    thresholds = {
        "category_accuracy": 0.78,
        "expected_kb_hit_at_5": 0.88,
        "citation_validity_rate": 0.98,
    }
    config.write_text(
        json.dumps(
            {
                "thresholds": thresholds,
                "retrieval_variant": "kb_focused",
                "dev_sha256": "frozen-dev",
            }
        )
    )
    calls = []
    result = report()

    def evaluate(settings, *, split, output):
        assert split == "dev" and not settings.llm_enabled
        assert not settings.solution_cache_enabled and settings.lexical_backend == "postgres"
        calls.append(output)
        return result

    monkeypatch.setattr(eval_gate, "evaluate_pipeline", evaluate)
    args = ["--thresholds", str(config), "--output", str(tmp_path / "result.json")]
    assert eval_gate.main(args) == 0
    result["summary"]["citation_contract_pass_rate"] = 0.97
    assert eval_gate.main(args) == 1
    assert calls == [tmp_path / "result.json"] * 2
    result["fingerprint"]["dev_sha256"] = "changed"
    with pytest.raises(SystemExit) as exc:
        eval_gate.main(args)
    assert exc.value.code == 2


def test_thresholds_are_baseline_minus_two_percentage_points():
    config = json.loads(Path("data/evaluation/gate_thresholds.json").read_text(encoding="utf-8"))
    for name in eval_gate.METRICS:
        assert config["thresholds"][name] == pytest.approx(max(0, config["baseline"][name] - 0.02))
