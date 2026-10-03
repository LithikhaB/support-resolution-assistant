"""Replay bounded customer turns without trusting client-supplied analysis or diagnoses."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.resolution.models import ResolutionRequest, ResolutionResponse
from app.understanding.context import clarification_questions, extract_facts
from app.understanding.models import AnalyzeRequest, ReportedFact, RuleAssessment, TextEvidence
from app.understanding.scope import service_group, split_issues
from app.understanding.service import get_understanding_service
from app.understanding.signals import assess_severity


class Observations(BaseModel):
    """Capture explicit customer answers, never provider diagnostic confirmation."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    wired_connection: Literal["working", "failing"] | None = None
    wireless_devices: Literal["one", "all"] | None = None
    impact: Literal["complete_loss", "intermittent", "working"] | None = None
    mobile_services: Literal["calls", "texts", "data", "several"] | None = None
    billing_status: Literal["pending", "settled"] | None = None
    charge: str | None = Field(default=None, min_length=1, max_length=100)
    tv_symptom: Literal["no_picture", "error", "buffering"] | None = None
    area: str | None = Field(default=None, min_length=1, max_length=100)
    started: str | None = Field(default=None, min_length=1, max_length=100)
    provider: str | None = Field(default=None, min_length=1, max_length=100)
    region: str | None = Field(default=None, min_length=1, max_length=100)


class CustomerTurn(BaseModel):
    """Attach a reply to exactly one issue; no assistant or system messages are accepted."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    issue_id: int = Field(ge=1, le=4, strict=True)
    message: str = Field(default="", max_length=1000)
    observations: Observations = Field(default_factory=Observations)

    @model_validator(mode="after")
    def require_answer(self):
        """Reject empty turns rather than counting them as an answered question."""
        if not self.message and not self.observations.model_dump(exclude_none=True):
            raise ValueError("a turn needs a message or an observation")
        return self


class ConversationRequest(ResolutionRequest):
    """Replay up to eight replies across at most four deterministically separated issues."""

    turns: list[CustomerTurn] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def validate_history(self):
        """Bound work and reject stale issue references before loading any models."""
        issues = split_issues(self.query)
        if len(issues) > 4:
            raise ValueError("submit at most four separate issues")
        if any(turn.issue_id > len(issues) for turn in self.turns):
            raise ValueError("reply issue_id does not exist in this complaint")
        for turn in self.turns:
            original = service_group(issues[turn.issue_id - 1])
            additional = service_group(turn.message)
            if original and additional - original:
                raise ValueError(
                    "a reply introduces another service; submit it as a separate issue"
                )
            if (
                turn.observations.wired_connection or turn.observations.wireless_devices
            ) and "home_connectivity" not in original:
                raise ValueError(
                    "wired and wireless observations require a home connectivity issue"
                )
            observations = turn.observations
            for present, group in (
                (observations.mobile_services, "mobile"),
                (observations.billing_status or observations.charge, "billing"),
                (observations.tv_symptom, "iptv"),
            ):
                if present and group not in original:
                    raise ValueError("observation does not match the selected issue's service")
        if len(self.model_dump_json()) > 16000:
            raise ValueError("conversation exceeds the history budget")
        for index, complaint in enumerate(issues, 1):
            text, _ = prepare_context(complaint, [t for t in self.turns if t.issue_id == index])
            if len(text) > 10000:
                raise ValueError("one issue exceeds the context budget")
        return self


class IssueResponse(BaseModel):
    """Keep each issue's evidence and offsets isolated from other reported problems."""

    issue_id: int
    complaint: str
    analysis_text: str
    resolution: ResolutionResponse


class ConversationResponse(BaseModel):
    """Return reviewable per-issue drafts without persisting private conversation history."""

    issues: list[IssueResponse]
    storage: Literal["client_replayed"] = "client_replayed"
    limitations: list[str] = Field(
        default_factory=lambda: [
            "Send the same original query and full customer-turn history on each request.",
            "Issue separation uses explicit service changes and does not recognize every compound complaint.",
            "Observations are customer reports, not verified diagnostic findings; no repairs or handoffs are executed.",
        ]
    )


def prepare_context(complaint, turns):
    """Preserve messages and append only the latest structured answer for each field."""
    text = ". ".join([complaint, *(t.message for t in turns if t.message)])
    latest = {}
    for turn in turns:
        for fact in extract_facts(turn.message):
            latest.pop(fact.name, None)
        if assess_severity(turn.message).value != "unknown":
            latest.pop("impact", None)
        latest.update(turn.observations.model_dump(exclude_none=True))
    facts = []
    for name, value in latest.items():
        statement = f"Customer reports {name.replace('_', ' ')}: {value}"
        start = len(text) + 2
        text += ". " + statement
        facts.append(
            ReportedFact(name=name, value=value, text=statement, start=start, end=len(text))
        )
    return text, facts


def apply_answers(analysis, text, turns, structured, *, history_start):
    """Use latest-turn observations while retaining attempts and unresolved contradictions."""
    boundaries = []
    cursor = history_start
    for turn in turns:
        if turn.message:
            cursor = text.index(turn.message, cursor)
            boundaries.append((cursor, cursor + len(turn.message)))
            current_impact = assess_severity(turn.message)
            if (
                current_impact.value != "unknown"
                and analysis.severity.rule != "reported_area_outage"
            ):
                for evidence in current_impact.evidence:
                    evidence.start += cursor
                    evidence.end += cursor
                analysis.severity = current_impact
            cursor += len(turn.message)
    facts = analysis.reported_facts
    for start, end in boundaries:
        updates = {f.name for f in facts if start <= f.start < end}
        facts = [f for f in facts if f.name not in updates or f.start >= start]
    names = {f.name for f in structured}
    analysis.reported_facts = [f for f in facts if f.name not in names] + structured
    impact = next((f for f in structured if f.name == "impact"), None)
    if impact and analysis.severity.rule != "reported_area_outage":
        analysis.severity = RuleAssessment(
            value={"complete_loss": "high", "intermittent": "medium", "working": "low"}[
                impact.value
            ],
            rule="customer_reported_current_impact",
            evidence=[TextEvidence(**impact.model_dump(include={"text", "start", "end"}))],
        )
    analysis.clarification_questions = clarification_questions(
        accepted=analysis.category is not None,
        products=analysis.products,
        severity=analysis.severity,
        facts=analysis.reported_facts,
        requests=analysis.requests,
    )
    known = {f.name: f.value for f in structured}
    if known.get("impact") == "working" and analysis.severity.rule != "reported_area_outage":
        analysis.clarification_questions = []
    if {"area", "started"} <= known.keys():
        analysis.clarification_questions = [
            q for q in analysis.clarification_questions if "shared outage" not in q
        ]
    if {"provider", "region"} <= known.keys():
        analysis.clarification_questions = [
            q for q in analysis.clarification_questions if "provider and country" not in q
        ]
    analysis.needs_clarification = bool(analysis.clarification_questions)
    if analysis.scope_status == "unsupported":
        analysis.clarification_questions = []
        analysis.needs_clarification = False
    return analysis


def resolve_conversation(request, service):
    """Analyze and retrieve each issue separately, replaying only that issue's replies."""
    understanding = service.understanding or get_understanding_service()
    results = []
    for issue_id, complaint in enumerate(split_issues(request.query), 1):
        turns = [t for t in request.turns if t.issue_id == issue_id]
        text, structured = prepare_context(complaint, turns)
        analysis = understanding.analyze(AnalyzeRequest(query=text))
        analysis = apply_answers(analysis, text, turns, structured, history_start=len(complaint))
        resolution = service.resolve(
            ResolutionRequest(
                query=text,
                max_sources=request.max_sources,
                rerank=request.rerank,
                filters=request.filters,
            ),
            analysis=analysis,
        )
        results.append(
            IssueResponse(
                issue_id=issue_id, complaint=complaint, analysis_text=text, resolution=resolution
            )
        )
    return ConversationResponse(issues=results)
