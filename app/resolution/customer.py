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
    if response.contact_status == "unverified":
        return CustomerPlan(
            title="Speak with human support",
            summary="You would like to speak with a person about this issue.",
            steps=[
                "Use the customer-care contact shown on your bill or in your service app. Share this complaint and the steps you have already tried."
            ],
            note="No call has been arranged by this assistant; it has no telephone directory.",
        )
    known = {(fact.name, fact.value) for fact in analysis.reported_facts}
    if ("bill_status", "unpaid") in known:
        return CustomerPlan(
            title="Your bill is still unpaid",
            summary="You said you plan to pay the bill. This is different from a payment already sent but still processing.",
            steps=[
                "Check the due date and amount in your bill or service app before paying.",
                "If you dispute the late fee, ask billing support to explain it against the due date and account payment record.",
            ],
            note="This assistant cannot access your account or approve a fee waiver.",
        )
    if response.clarification_questions:
        title = "One more detail"
        if analysis.category:
            title = f"Investigating {analysis.category.replace('_', ' ').title()}"

        summary = "Please answer below so I can suggest the next step."
        if ("charge", "late_fee") in known:
            summary = "You are asking about a late fee. Payment timing is needed to explain what billing support should check."
        elif response.acknowledged_actions:
            summary = f"Noted that you have already tried: {'; '.join(response.acknowledged_actions)}. To confirm the exact resolution, please check the detail requested below."
        elif analysis.category:
            summary = f"Initial assessment indicates {analysis.category.replace('_', ' ')}. Please review the initial checks and answer the question below."

        steps = []
        if response.suggestions and response.validation.status != "failed":
            for item in response.suggestions:
                if item.repeated_actions:
                    steps.append(
                        f"[{item.citation_id}] You have already tried an action in this procedure. Share its outcome with support instead of repeating it. Next diagnostic check: {item.required_finding}"
                    )
                else:
                    steps.append(
                        f"[{item.citation_id}] Ask support to check: {item.required_finding} Only if confirmed: {item.proposed_action} {item.restriction}"
                    )
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
            note="Provisional guidance while we verify details. Answering below helps select the right procedure.",
        )
    if response.suggestions and response.validation.status != "failed":
        steps = []
        for item in response.suggestions:
            if item.repeated_actions:
                steps.append(
                    f"[{item.citation_id}] You have already tried an action in this procedure. Share its result with support instead of repeating it. The next check is: {item.required_finding}"
                )
            else:
                steps.append(
                    f"[{item.citation_id}] Ask support to check: {item.required_finding} If confirmed, the documented next step is: {item.proposed_action} {item.restriction}"
                )
        return CustomerPlan(
            title="Your next step",
            summary="The relevant procedure needs a support check before a repair can be selected.",
            steps=steps,
            note="Based on the demo knowledge base. No repair has been performed.",
        )
    return CustomerPlan(
        title="This ticket needs investigation",
        summary="The available information does not support a specific repair yet.",
        steps=[
            "Contact support with the symptoms and checks you have already tried so they can investigate."
        ],
        note="No external handoff has been created.",
    )
