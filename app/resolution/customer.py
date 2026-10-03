"""Present a clear ticket resolution plan independently of internal diagnostic details."""

from app.resolution.models import CustomerPlan


def physical_damage(analysis):
    """Identify explicit reported equipment damage without inferring its cause."""
    return any(
        f.name == "equipment_condition" and f.value in {"damaged", "water_exposed"}
        for f in analysis.reported_facts
    )


def customer_plan(response):
    """Translate workflow state into honest next steps without exposing diagnostic speculation."""
    analysis = response.analysis
    if analysis.scope_status == "unsupported":
        return CustomerPlan(
            title="This needs another support service",
            summary="This assistant handles telecom complaints. We cannot recommend a repair for this request.",
            steps=["Contact the organization responsible for the affected service."],
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
            title="Your damaged equipment needs an inspection",
            summary="You reported physical damage to your telecom equipment. Even if service in the area has recovered, that does not establish that your equipment is safe or usable.",
            steps=steps,
            note="A replacement has not been booked. Your provider must confirm eligibility, charges and timing.",
        )
    if any(
        f.name in {"impact", "service_recovery"} and f.value == "working"
        for f in analysis.reported_facts
    ):
        return CustomerPlan(
            title="You reported that service is working again",
            summary="No further repair is recommended from the information provided.",
            steps=["Monitor the connection and record when the issue returns, if it does."],
            note="An agent can review the outcome before closing the case.",
        )
    if analysis.severity.rule == "reported_area_outage":
        return CustomerPlan(
            title="Report the shared outage to your provider",
            summary="Several people or locations are affected. This needs a provider incident check rather than repeated device resets.",
            steps=[
                "Ask your provider whether a network incident is open for your area and request an update reference."
            ],
            note="This assistant has not checked live network status or opened a provider ticket.",
        )
    if response.contact_status == "unverified" and not analysis.products:
        return CustomerPlan(
            title="Arrange a human support follow-up",
            summary="The customer wants to speak with support. Route the request to the support desk.",
            steps=[
                "Record the call preference and arrange a follow-up through the existing support process."
            ],
            note="No call has been arranged by this assistant; it has no telephone directory.",
        )
    if response.clarification_questions:
        return CustomerPlan(
            title="Let's narrow down the problem",
            summary="We need the details below to choose the next step without guessing a cause.",
            steps=["Answer the questions below together in the reply box."],
            note="Your previous attempts are retained so they can be reviewed.",
        )
    if response.suggestions and response.validation.status != "failed":
        steps = []
        for item in response.suggestions:
            if item.repeated_actions:
                steps.append(
                    f"[{item.citation_id}] Review the result of the earlier attempt; do not repeat it automatically. Check: {item.required_finding}"
                )
            else:
                steps.append(
                    f"[{item.citation_id}] Verify: {item.required_finding} Only if confirmed: {item.proposed_action} Restriction: {item.restriction}"
                )
        return CustomerPlan(
            title="Suggested resolution for agent review",
            summary="These steps depend on the diagnostic checks shown below. Similar symptoms alone do not confirm the cause.",
            steps=steps,
            note="Sources are synthetic demonstration procedures. No repair has been executed.",
        )
    return CustomerPlan(
        title="This ticket needs investigation",
        summary="No applicable resolution is supported by the available evidence. Route the ticket to an agent instead of guessing a repair.",
        steps=[
            "Review the reported symptoms and previous attempts, then obtain the missing diagnostic evidence."
        ],
        note="No external handoff has been created.",
    )
