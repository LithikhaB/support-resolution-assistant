"""Regression checks for agent plans, exact parent evidence and quota backoff."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from app.config.settings import Settings
from app.llm.contracts import LanguageUnavailable
from app.llm.providers import ProviderChain
from app.resolution.applicability import applicability_issue
from app.resolution.conversation import CustomerTurn, apply_answers, prepare_context
from app.resolution.drafting import draft_resolution
from app.resolution.evidence import parse_procedure, supported_scopes
from app.resolution.selection import rank_fallback
from app.resolution.validation import finalize_resolution, validate_citations
from app.understanding.context import extract_facts
from app.understanding.signals import assess_sentiment, assess_severity, extract_actions
from scripts.enrich_procedures import enrich
from scripts.evaluate_resolution_path import audit_examples
from tests.unit.test_resolution import analysis, kb

STORM = "Since the storm on Tuesday my internet cuts out for about a minute every hour or so. The router light stays green. I unplugged it for ten minutes, twice. I run a small online tailoring shop so it's costing me orders."
OPTICAL = "No internet at all. I restarted the router three times, checked every cable, reset it to factory settings and waited an hour. The optical box has a red light."


def test_optical_alarm_preserves_completed_checks_and_rejects_local_port_repair():
    from app.resolution.policy import questions_for

    observed = analysis(OPTICAL)
    assert {a.action for a in observed.actions} >= {
        "restart_device",
        "check_cables",
        "reset_settings",
    }
    assert any("three times" in a.text for a in observed.actions)
    assert questions_for(observed) == [
        "Is the red indicator labelled LOS, and when did service stop working?"
    ]
    source = kb().model_copy(
        update={
            "content": kb().content.replace(
                "An authorized line test confirms the fault.",
                "Port diagnostics confirm a physical port fault; service and other ports are healthy.",
            )
        }
    )
    assert applicability_issue(parse_procedure(source), observed) == (
        "defer_other_procedures_until_reported_optical_alarm_checked"
    )


def test_alternative_procedures_have_one_final_verification_step():
    source = enriched_source()
    second = source.model_copy(
        update={
            "doc_id": "second",
            "chunk_id": 2,
            "evidence_content": source.evidence_content.replace(
                "Raise a fibre repair job", "Arrange an authorized fibre inspection"
            ),
        }
    )
    response = draft_resolution(analysis("My internet drops."), [source, second])
    checks = [s for s in response.customer_plan.steps if s.startswith("After any authorized")]
    assert len(checks) == 1
    assert "[S1] [S2]" in checks[0]
    assert not any("simulation completion criteria" in s for s in response.customer_plan.steps)


@pytest.mark.parametrize(
    "text",
    [
        STORM,
        "My broadband drops every evening around 8 and I've already restarted the router twice, I work from home and this is costing me.",
    ],
)
def test_business_impact_and_attempt_counts_have_exact_evidence(text):
    assert assess_severity(text).value == "high"
    assert assess_sentiment(text).value == "frustrated"
    actions = extract_actions(text)
    assert "twice" in actions[0].text
    for evidence in [*actions, *assess_severity(text).evidence, *assess_sentiment(text).evidence]:
        assert text[evidence.start : evidence.end] == evidence.text


def enriched_source():
    row = json.loads(
        Path("data/synthetic/telecom_v1/knowledge_base.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    row = enrich(row)
    return kb().model_copy(
        update={
            "content": "Partial searchable chunk",
            "evidence_content": row["body"],
            "metadata": row["metadata"],
        }
    )


def test_parent_procedure_is_readable_and_every_quote_is_validated():
    source = enriched_source()
    observed = analysis("My internet drops. I already restarted the router twice.")
    response = finalize_resolution(draft_resolution(observed, [source]), [source], observed)
    assert response.validation.status == "passed"
    assert response.sources[0].quote_scope == "parent_document"
    assert len(response.customer_plan.steps) >= 4
    assert response.customer_plan.steps[0].startswith("Do not repeat")
    assert any("Only if support confirms" in s for s in response.customer_plan.steps)
    assert not any(s.startswith("Support check:") for s in response.customer_plan.steps)
    for quote in response.sources[0].quotes:
        assert source.evidence_content[quote.start : quote.end] == quote.text
    response.sources[0].quotes[-1].text = "Invented completion"
    assert validate_citations(response, [source], observed).status == "failed"


def test_incomplete_v2_procedure_fails_closed():
    source = enriched_source()
    source.evidence_content = source.evidence_content.split("Completion:")[0]
    assert parse_procedure(source) is None


def test_weather_is_a_signal_and_keeps_power_scope_without_claiming_damage():
    observed = analysis(STORM)
    assert "router" in supported_scopes(observed)
    assert any(f.name == "weather_context" for f in extract_facts(STORM))
    source = kb().model_copy(
        update={
            "content": kb().content.replace(
                "An authorized line test confirms the fault.",
                "Peak-hour capacity congestion is confirmed.",
            )
        }
    )
    assert (
        applicability_issue(parse_procedure(source), observed)
        == "peak_hour_gate_without_peak_timing_in_weather_report"
    )


def test_rate_limit_backoff_is_nonblocking_exponential_and_respects_retry_after():
    clock = Mock(return_value=0)
    provider = Mock(name="provider")
    provider.name = "groq"
    provider.generate.side_effect = LanguageUnavailable("provider_http_429", retry_after=90)
    chain = ProviderChain(
        Settings(_env_file=None, llm_circuit_seconds=60), providers=[provider], clock=clock
    )
    for now, deadline in [(0, 90), (91, 211), (212, 452)]:
        clock.return_value = now
        with pytest.raises(LanguageUnavailable):
            chain.generate("instruction", {}, Settings)
        assert chain.circuits["groq"] == deadline
    with pytest.raises(LanguageUnavailable, match="circuit_open"):
        chain.generate("instruction", {}, Settings)
    assert provider.generate.call_count == 3


def test_current_fewshots_are_uniquely_train_family_only():
    assert audit_examples(Path("data/synthetic/telecom_v1"))["heldout_example_count"] == 0


@pytest.mark.parametrize("structured", [False, True])
def test_followup_does_not_erase_reported_business_impact(structured):
    turn = CustomerTurn(
        issue_id=1,
        message="The wired connection still drops.",
        observations={"impact": "intermittent"} if structured else {},
    )
    text, facts = prepare_context(STORM, [turn])
    result = apply_answers(analysis(text), text, [turn], facts, history_start=len(STORM))
    assert result.severity.value == "high"
    assert result.severity.rule == "reported_business_impact"


@pytest.mark.parametrize(
    "text,first", [(STORM, "syn_kb_ID01"), ("My broadband drops every evening.", "syn_kb_ID02")]
)
def test_explicit_context_orders_investigations_without_confirming_findings(text, first):
    observed = analysis(text)
    observed.category = "intermittent_broadband"
    observed.category_basis = "explicit_report"
    rows = [
        json.loads(line)
        for line in Path("data/synthetic/telecom_v1/knowledge_base.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    evidence = [
        kb().model_copy(
            update={
                "chunk_id": i,
                "doc_id": row["doc_id"],
                "content": row["body"],
                "metadata": row["metadata"],
            }
        )
        for i, row in enumerate(rows)
    ]
    ranked = rank_fallback(evidence, observed)
    assert ranked[0].doc_id == first
    response = draft_resolution(observed, ranked, max_sources=2)
    assert all(s.status == "requires_agent_confirmation" for s in response.suggestions)
    assert response.agent_review_required


def test_duplicate_checks_coalesce_with_all_supporting_citations():
    source = enriched_source()
    second = source.model_copy(
        update={
            "doc_id": "second",
            "chunk_id": 2,
            "evidence_content": source.evidence_content.replace(
                "Raise a fibre repair job", "Arrange an authorized fibre inspection"
            ),
        }
    )
    response = draft_resolution(analysis("My internet drops."), [source, second])
    shared = [step for step in response.customer_plan.steps if step.startswith("Ask whether")]
    assert len(shared) == 1
    assert "[S1] [S2]" in shared[0]


@pytest.mark.parametrize(
    "gate",
    [
        "Gateway logs show session renewals failing against a stale lease record.",
        "Optical signal is healthy; the subscriber session is rejected after an account migration.",
        "Line stays synchronized; provider monitoring shows peak-hour congestion on the access segment.",
    ],
)
def test_reported_optical_alarm_defers_session_and_capacity_procedures(gate):
    source = kb().model_copy(
        update={
            "content": kb().content.replace("An authorized line test confirms the fault.", gate)
        }
    )
    assert applicability_issue(parse_procedure(source), analysis(OPTICAL)) == (
        "defer_other_procedures_until_reported_optical_alarm_checked"
    )


def test_compact_plan_keeps_gate_action_completion_and_escalation():
    source = enriched_source()
    response = draft_resolution(analysis(OPTICAL), [source])
    procedure = parse_procedure(source)
    text = " ".join(response.customer_plan.steps)
    assert procedure.quotes["condition"].text in text
    assert procedure.quotes["action"].text in text
    assert procedure.quotes["completion"].text in text
    assert "Only if support confirms" in text
    assert "refer for specialist review" in response.customer_plan.note
    assert "Record the affected service" not in text


def test_generated_wording_keeps_completed_actions_and_source_completion():
    from app.resolution.language import (
        DraftIntroduction,
        FaithfulnessReview,
        TroubleshootingStep,
        add_language_draft,
    )

    response = draft_resolution(analysis(OPTICAL), [enriched_source()])
    client = Mock()
    client.name = "groq"
    client.generate.side_effect = [
        DraftIntroduction(
            summary="Customer reports an outage and completed checks. Verify the optical signal before a repair.",
            history_citations=[],
            procedure_citations=["S1"],
            steps=[
                TroubleshootingStep(
                    instruction="Ask authorized support to verify the optical signal and account status before considering a fibre repair.",
                    citation_id="S1",
                )
            ],
        ),
        FaithfulnessReview(supported=True, issues=[]),
    ]
    result = add_language_draft(
        response, OPTICAL, Settings(_env_file=None, llm_enabled=True), client
    )
    assert result.language_status == "generated_for_review"
    assert result.language_plan.steps[0] == response.customer_plan.steps[0]
    assert result.language_plan.steps[-1] == response.customer_plan.steps[-1]
    payload = client.generate.call_args_list[0].args[1]
    assert (
        payload["source_procedures"][0]["quotes"]["condition"]
        == response.suggestions[0].required_finding
    )
    assert "refer for specialist review" in result.language_plan.note
