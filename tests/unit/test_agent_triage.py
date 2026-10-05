"""Regression checks for agent plans, exact parent evidence and quota backoff."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from app.config.settings import Settings
from app.llm.contracts import LanguageUnavailable
from app.llm.providers import ProviderChain
from app.resolution.applicability import applicability_issue
from app.resolution.drafting import draft_resolution
from app.resolution.evidence import parse_procedure
from app.resolution.selection import rank_fallback
from app.resolution.validation import finalize_resolution, validate_citations
from app.understanding.signals import assess_sentiment, assess_severity, extract_actions
from scripts.enrich_procedures import enrich
from tests.unit.test_resolution import analysis, kb

STORM = "Since the storm on Tuesday my internet cuts out for about a minute every hour or so. The router light stays green. I unplugged it for ten minutes, twice. I run a small online tailoring shop so it's costing me orders."
OPTICAL = "No internet at all. I restarted the router three times, checked every cable, reset it to factory settings and waited an hour. The optical box has a red light."


def test_optical_alarm_preserves_completed_checks_and_rejects_local_port_repair():
    fee = analysis("My bill has a late fee, but every service works.")
    assert (
        applicability_issue(parse_procedure(kb(scope="billing")), fee)
        == "unrelated_procedure_for_reported_late_fee"
    )
    from app.resolution.policy import questions_for

    wireless = analysis("Web keeps vanishing around dinner time my wired PC stays online.")
    profile = kb(scope="home_wifi").model_copy(
        update={
            "content": kb(scope="home_wifi").content.replace(
                "An authorized line test confirms the fault.",
                "Laptop retains an old saved wireless profile; other clients authenticate.",
            )
        }
    )
    assert applicability_issue(parse_procedure(profile), wireless) == (
        "authentication_gate_for_reported_wireless_drops"
    )

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
    paid = analysis("My bill shows two settled payments for the same invoice.")
    for finding, expected in (
        ("The ledger confirms two settled payments against the same invoice.", None),
        (
            "The payment was allocated to a different account.",
            "different_payment_issue_for_reported_duplicate_settlement",
        ),
        (
            "The ledger shows correct prorated periods and no duplicate charge.",
            "different_payment_issue_for_reported_duplicate_settlement",
        ),
    ):
        billing = kb(scope="billing").model_copy(
            update={
                "content": kb(scope="billing").content.replace(
                    "An authorized line test confirms the fault.", finding
                )
            }
        )
        assert applicability_issue(parse_procedure(billing), paid) == expected


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


def v3_source():
    row = json.loads(
        Path("data/synthetic/telecom_v3/knowledge_base.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    return kb().model_copy(
        update={
            "doc_id": row["doc_id"],
            "title": row["title"],
            "content": "Searchable chunk",
            "evidence_content": row["body"],
            "metadata": row["metadata"],
        }
    )


def test_parent_procedure_is_readable_and_every_quote_is_validated():
    from app.resolution.customer import customer_plan
    from app.resolution.history import select_history

    observed = analysis("My internet drops. I already restarted the router twice.")
    for source in (enriched_source(), v3_source()):
        response = finalize_resolution(draft_resolution(observed, [source]), [source], observed)
        assert response.validation.status == "passed"
        assert response.sources[0].quote_scope == "parent_document"
        assert len(response.customer_plan.steps) >= 4
        assert response.customer_plan.steps[0].startswith("Do not repeat")
        assert any("Only if support confirms" in s for s in response.customer_plan.steps)
        for quote in response.sources[0].quotes:
            assert source.evidence_content[quote.start : quote.end] == quote.text
        if source.metadata.get("procedure_version") == 3:
            row = json.loads(
                Path("data/synthetic/telecom_v3/tickets.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            ticket = kb().model_copy(
                update={
                    "doc_id": row["doc_id"],
                    "chunk_id": 2,
                    "doc_type": "resolved_ticket",
                    "resolution": row["resolution"],
                    "outcome_status": row["outcome_status"],
                    "metadata": row["metadata"],
                }
            )
            response.historical_cases = select_history([ticket], response.sources)
            response.customer_plan = customer_plan(response)
            assert any("[S1] [T1]" in s for s in response.customer_plan.steps)
            assert (
                validate_citations(
                    response, [source], observed, historical_evidence=[ticket]
                ).status
                == "passed"
            )
            response.historical_cases[0].resolution_steps[0].text = "Invented step"
            assert (
                validate_citations(
                    response, [source], observed, historical_evidence=[ticket]
                ).status
                == "failed"
            )
            response.historical_cases = []
            response.customer_plan = customer_plan(response)
        response.sources[0].quotes[-1].text = "Invented completion"
        assert validate_citations(response, [source], observed).status == "failed"


def test_incomplete_procedures_and_ungated_v3_steps_fail_closed():
    for source in (enriched_source(), v3_source()):
        source.evidence_content = source.evidence_content.split("Completion:")[0]
        assert parse_procedure(source) is None
    source = v3_source()
    assert parse_procedure(source) is not None
    source.metadata["procedure"]["steps"][3]["text"] = "Repair the line now."
    assert parse_procedure(source) is None
    from app.resolution.memory import history_context
    from app.resolution.models import ResolutionRequest
    from app.resolution.reuse import SemanticReuse

    clock = Mock(return_value=0)
    reuse = SemanticReuse(seconds=5, clock=clock)
    observed = analysis("My broadband is down.")
    observed.category, observed.scope_status = "broadband_outage", "supported"
    request = ResolutionRequest(query="My broadband is down.")
    settings = Settings(_env_file=None)
    token = history_context.set(("browser-a", None, ("v1",)))
    try:
        signature = reuse.signature(observed, request, ("v1",), settings)
        vector = [1.0] + [0.0] * 383
        reuse.put(signature, vector, [v3_source()])
        assert reuse.get(signature, vector, 0.98)
        assert reuse.get(signature, [0.0, 1.0] + [0.0] * 382, 0.98) is None
        assert reuse.signature(observed, request, ("v2",), settings) != signature
        assert (
            reuse.signature(
                observed, request.model_copy(update={"max_sources": 1}), ("v1",), settings
            )
            != signature
        )
        history_context.set(("browser-b", None, ("v1",)))
        assert reuse.signature(observed, request, ("v1",), settings) != signature
        clock.return_value = 6
        assert reuse.get(signature, vector, 0.98) is None
    finally:
        history_context.reset(token)
    source = v3_source()
    source.evidence_content += "\nStep 1: Duplicate instruction"
    assert parse_procedure(source) is None


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


def test_generated_wording_keeps_completed_actions_and_source_completion():
    from app.resolution.language import (
        DraftIntroduction,
        FaithfulnessReview,
        PlanWording,
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

    # An incomplete generated v3 plan must fall back, even if its summary is plausible.
    response = draft_resolution(analysis(OPTICAL), [v3_source()])
    client.generate.side_effect = [
        DraftIntroduction(
            summary="Customer reports an outage. Verify the optical signal before a repair.",
            history_citations=[],
            procedure_citations=["S1"],
            steps=[
                TroubleshootingStep(
                    instruction="Ask authorized support to verify optical signal before repair.",
                    citation_id="S1",
                )
            ],
        ),
        FaithfulnessReview(supported=True, issues=[]),
    ]
    result = add_language_draft(
        response, OPTICAL, Settings(_env_file=None, llm_enabled=True), client
    )
    assert result.language_status == "fallback"
    assert "up to four short steps" not in client.generate.call_args_list[-1].args[0]
    assert result.language_error == "incomplete_ordered_plan"
    assert result.language_plan is None
    source_payload = client.generate.call_args_list[-1].args[1]["source_procedures"][0]
    assert all(f"step_{i}" in source_payload["quotes"] for i in range(1, 6))
    original = response.customer_plan.steps
    proposal = DraftIntroduction(
        summary="Customer reports no internet after restarting the router.",
        history_citations=[],
        procedure_citations=["S1"],
        plan_steps=[
            PlanWording(original_index=i, instruction=step) for i, step in enumerate(original, 1)
        ],
    )
    # Only an investigation check is reworded; gate, action, completion and prior actions stay exact.
    editable = next(i for i, step in enumerate(original) if "Ask which light" in step)
    proposal.plan_steps[editable].instruction = original[editable].replace(
        "Ask which light", "Record which light"
    )
    # Citations are attached by the application, so omission cannot lose KB/history evidence.
    import re

    proposal.plan_steps[editable].instruction = re.sub(
        r"\[[ST]\d+\]", "", proposal.plan_steps[editable].instruction
    ).strip()
    client.generate.side_effect = [proposal, FaithfulnessReview(supported=True, issues=[])]
    response.language_status = "disabled"
    result = add_language_draft(
        response, OPTICAL, Settings(_env_file=None, llm_enabled=True), client
    )
    assert result.language_status == "generated_for_review"
    assert result.language_plan.steps[editable] != original[editable]
    assert result.language_plan.steps[-2:] == original[-2:]
    assert result.language_plan.note == result.customer_plan.note
    for kind in ("gate", "citation", "order"):
        rejected = proposal.model_copy(deep=True)
        if kind == "gate":
            rejected.plan_steps[-2].instruction = "Replace the fibre immediately. [S1]"
        elif kind == "citation":
            rejected.plan_steps[editable].instruction += " [S2]"
        else:
            rejected.plan_steps.reverse()
        client.generate.side_effect = [rejected]
        result = add_language_draft(
            response.model_copy(deep=True, update={"language_plan": None}),
            OPTICAL,
            Settings(_env_file=None, llm_enabled=True),
            client,
        )
        assert result.language_status == "fallback" and result.language_plan is None
