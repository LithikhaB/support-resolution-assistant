"""Present a clear ticket resolution plan independently of internal diagnostic details."""

from app.resolution.models import CustomerPlan
from app.understanding.signals import extract_actions


def physical_damage(analysis):
    """Identify explicit reported equipment damage without inferring its cause."""
    return any(
        f.name == "equipment_condition" and f.value in {"damaged", "water_exposed"}
        for f in analysis.reported_facts
    )


def procedure_steps(response):
    """Render ordered source checks and keep all remedies conditional on the full gate."""
    if response.sources and all(
        any(q.field == "plan_step" for q in source.quotes) for source in response.sources
    ):
        return ordered_procedure_steps(response)
    steps = []
    if response.acknowledged_actions:
        steps.append(
            "Do not repeat completed checks: " + "; ".join(response.acknowledged_actions) + "."
        )
    attempted = {a.action for a in response.analysis.actions if a.status == "attempted"}
    known = {f.name for f in response.analysis.reported_facts}
    optical_alarm = any(
        f.name == "optical_signal" and f.value == "loss_reported"
        for f in response.analysis.reported_facts
    )
    # Coalesce identical checks across alternative procedures, retaining every citation.
    shared = {}
    completion = {}
    for item in response.suggestions:
        source = next((s for s in response.sources if s.citation_id == item.citation_id), None)
        fields = {q.field: q.text for q in source.quotes} if source else {}
        for field in ("customer_checks",):
            text = fields.get(field)
            if not text or (
                field == "customer_checks"
                and (
                    (("wired_connection" in known or optical_alarm) and "wired device" in text)
                    or response.analysis.severity.rule == "reported_area_outage"
                    or ("billing_status" in known and "pending or settled" in text)
                )
            ):
                continue
            if attempted & {a.action for a in extract_actions(text)}:
                continue
            shared.setdefault(text, []).append(item.citation_id)
        if "agent_checks" in fields:
            steps.append(
                f"Check provider diagnostics: {item.required_finding} [{item.citation_id}]"
            )
        if "agent_checks" not in fields:
            steps.append(
                f"Check with authorized provider diagnostics whether all of the following are established: {item.required_finding} [{item.citation_id}]"
            )
        if item.repeated_actions:
            steps.append(
                f"Withhold the previously attempted remedy and review its outcome. [{item.citation_id}]"
            )
        else:
            steps.append(
                f"Only if support confirms the full diagnostic gate: {item.proposed_action} [{item.citation_id}]"
            )
        if "completion" in fields:
            completion.setdefault(fields["completion"], []).append(item.citation_id)
    common = [
        f"{text} " + " ".join(f"[{c}]" for c in citations) for text, citations in shared.items()
    ]
    ending = []
    if completion:
        ending.insert(
            0,
            "After any authorized repair, verify the relevant result: "
            + "; ".join(
                f"{text} " + " ".join(f"[{c}]" for c in citations)
                for text, citations in completion.items()
            ),
        )
    prefix = steps[:1] if response.acknowledged_actions else []
    return prefix + common + steps[len(prefix) :] + ending


def ordered_procedure_steps(response):
    """Use exact v3 step spans; attach a ticket citation only to an exact shared step."""
    steps = []
    if response.acknowledged_actions:
        steps.append(
            "Do not repeat completed checks: " + "; ".join(response.acknowledged_actions) + "."
        )
    known = {f.name for f in response.analysis.reported_facts}
    attempted = {a.action for a in response.analysis.actions if a.status == "attempted"}
    completion = []
    seen = {}
    for index, source in enumerate(response.sources):
        suggestion = next(s for s in response.suggestions if s.citation_id == source.citation_id)
        for quote in sorted(
            (q for q in source.quotes if q.field == "plan_step"), key=lambda q: q.step_id
        ):
            if quote.phase == "conditional_fix" and suggestion.repeated_actions:
                continue
            if quote.skip_if_fact in known:
                continue
            if quote.phase == "customer_check" and attempted & {
                a.action for a in extract_actions(quote.text)
            }:
                continue
            # The primary procedure provides the investigation sequence. Alternatives
            # contribute a fully gated remedy, rather than another complete checklist.
            if index and quote.phase not in {"conditional_fix", "completion"}:
                continue
            citations = [source.citation_id]
            for case in response.historical_cases:
                if (
                    case.relationship == "linked_procedure"
                    and source.doc_id in case.kb_refs
                    and any(
                        h.text == quote.text and case.resolution[h.start : h.end] == h.text
                        for h in case.resolution_steps
                    )
                ):
                    citations.append(case.citation_id)
                    break
            displayed = quote.text
            if (
                quote.phase == "customer_check"
                and displayed.startswith("Ask which light on the fibre box is red:")
                and any(
                    f.name == "optical_signal" and "los" in f.text.lower()
                    for f in response.analysis.reported_facts
                )
            ):
                displayed = displayed[displayed.index("Record whether") :]
            text = displayed + " " + " ".join(f"[{c}]" for c in citations)
            if quote.phase == "completion":
                completion.append(text)
            elif quote.text not in seen:
                seen[quote.text] = len(steps)
                steps.append(text)
    if completion:
        steps.append(" ".join(completion))
    return steps


def procedure_note(response):
    restrictions = " ".join(
        f"{item.restriction} [{item.citation_id}]"
        for item in response.suggestions
        if "customer is angry" not in item.restriction.lower()
        or response.analysis.sentiment.value == "angry"
    )
    escalation = {}
    for source in response.sources:
        for quote in source.quotes:
            if quote.field == "escalate_if":
                escalation.setdefault(quote.text, []).append(source.citation_id)
    for text, citations in escalation.items():
        if text == (
            "If the gate is unconfirmed, contradictory or outside the agent's authority, "
            "route for specialist investigation with the observations; do not apply the remedy."
        ):
            text = "If a required finding is unconfirmed, contradictory or outside your authority, refer for specialist review; do not apply the remedy."
        restrictions += " " + text + " " + " ".join(f"[{c}]" for c in citations)
    linked = [h for h in response.historical_cases if h.relationship == "linked_procedure"]
    if linked:
        restrictions += (
            " Compare the linked simulated outcomes "
            + " ".join(f"[{h.citation_id}]" for h in linked)
            + "; they inform investigation but do not confirm this customer's cause."
        )
    return restrictions


def customer_plan(response):
    """Translate workflow state into honest next steps without exposing diagnostic speculation."""
    analysis = response.analysis
    if any(
        f.name == "security_request" and f.value == "otp_sharing" for f in analysis.reported_facts
    ):
        return CustomerPlan(
            title="Keep your verification code private",
            summary="A request to read out a login or verification code could put your account at risk.",
            steps=[
                "Do not share the code or your password with the caller.",
                "End the contact and verify the request through the support contact in your provider's official app or bill.",
            ],
            note="Support should review the request. This assistant cannot verify who called you.",
        )
    if analysis.scope_status == "unsupported":
        return CustomerPlan(
            title="This request is irrelevant here",
            summary="This assistant handles telecom service and billing complaints. Please paste the customer's actual complaint, including the affected service and symptoms.",
            steps=[],
            note="General-knowledge questions and requests to generate a complaint are outside this assistant's scope.",
        )
    if physical_damage(analysis):
        water = any(
            f.name == "equipment_condition" and f.value == "water_exposed"
            for f in analysis.reported_facts
        )
        steps = [
            "Route this ticket for a technician inspection and replacement assessment of the damaged equipment and line.",
            "Record which equipment is damaged and request an inspection estimate. The support team must confirm replacement eligibility and timing.",
        ]
        if water:
            steps.insert(
                0,
                "Do not use or power on wet equipment. Keep clear of wet electrical connections and have a qualified professional inspect them before use.",
            )
        return CustomerPlan(
            title="Reported equipment damage requires inspection",
            summary="The customer reported physical damage. Area service recovery does not establish that this equipment is safe or usable.",
            steps=steps,
            note="A replacement has not been booked. Your provider must confirm eligibility, charges and timing.",
        )
    if {p.product for p in analysis.products} == {"landline"}:
        return CustomerPlan(
            title="Contact landline support",
            summary="You reported a landline fault. The current knowledge base has no supported landline repair procedure.",
            steps=[
                "Route this issue to the fixed-line support team with the symptoms and previous attempts."
            ],
            note="No repair or external handoff has been performed.",
        )
    if "billing" not in {p.product for p in analysis.products} and any(
        f.name in {"impact", "service_recovery"} and f.value == "working"
        for f in analysis.reported_facts
    ):
        return CustomerPlan(
            title="Customer reports service recovered",
            summary="No further repair is recommended from the information provided.",
            steps=["Monitor the connection and record when the issue returns, if it does."],
            note="An agent can review the outcome before closing the case.",
        )
    if analysis.severity.rule == "reported_area_outage":
        return CustomerPlan(
            title="Review the shared outage as a provider incident",
            summary="Several people or locations are affected. This needs a provider incident check rather than repeated device resets.",
            steps=procedure_steps(response)
            if response.suggestions
            else [
                "Check authorized provider incident monitoring for the affected area and record an update reference."
            ],
            note="This assistant has not checked live network status or opened a provider ticket.",
        )
    if response.contact_status == "unverified":
        return CustomerPlan(
            title="Route to human support",
            summary="The customer requested human support for this issue.",
            steps=[
                "Use the customer-care contact shown on your bill or in your service app. Share this complaint and the steps you have already tried."
            ],
            note="No call has been arranged by this assistant; it has no telephone directory.",
        )
    known = {(fact.name, fact.value) for fact in analysis.reported_facts}
    if ("bill_status", "unpaid") in known:
        return CustomerPlan(
            title="Customer reports an unpaid bill",
            summary="The customer plans to pay the bill. Verify the account record before treating this as a sent payment awaiting settlement.",
            steps=[
                "Check the due date and amount in your bill or service app before paying.",
                "If you dispute the late fee, ask billing support to explain it against the due date and account payment record.",
            ],
            note="This assistant cannot access your account or approve a fee waiver.",
        )
    if response.clarification_questions:
        title = "One more detail"
        if analysis.category:
            title = "Initial checks"

        summary = "Record the customer's answer to the focused question below, then review the conditional investigation."
        if ("charge", "late_fee") in known:
            summary = "You are asking about a late fee. Payment timing is needed to explain what billing support should check."
        elif response.acknowledged_actions:
            summary = "Review completed checks and their outcomes. Record the customer's answer before authorizing a remedy."
        elif analysis.category:
            summary = "Review the initial check below. Your answer will help narrow the next step."

        steps = []
        if response.suggestions and response.validation.status != "failed":
            steps = procedure_steps(response)
        elif analysis.category == "intermittent_broadband":
            steps = [
                "Observe whether drops affect all connected devices or only wireless connections.",
                "Check the status lights on your modem/router (power, internet, Wi-Fi) during a drop.",
            ]
        elif analysis.category == "wifi_connectivity":
            steps = [
                "Check whether Wi-Fi drops occur only in specific rooms or across the whole property.",
                "Verify whether any other device remains connected when a disconnection occurs.",
            ]
        elif analysis.category == "broadband_outage":
            steps = [
                "Check whether the optical (LOS/PON) or broadband light on your router/ONT is illuminated.",
                "Verify if other connected devices or neighbours on the same line are offline.",
            ]
        if not steps:
            steps = [
                "Note when the problem started, which services or devices are affected, and any error message you see.",
                "Share the checks you have already tried and their results with support so you do not have to repeat them.",
            ]
        return CustomerPlan(
            title=title,
            summary=summary,
            steps=steps,
            note=procedure_note(response)
            if response.suggestions
            else "These are initial checks while we confirm the details.",
        )
    if response.suggestions and response.validation.status != "failed":
        return CustomerPlan(
            title="Recommended next steps",
            summary="Start with the check below. Use the suggested action only after the finding is confirmed.",
            steps=procedure_steps(response),
            note=procedure_note(response),
        )
    return CustomerPlan(
        title="This ticket needs investigation",
        summary="The available information does not support a specific repair yet.",
        steps=[
            "Contact support with the symptoms and checks you have already tried so they can investigate."
        ],
        note="No external handoff has been created.",
    )
