"""Generate a synthetic telecom support dataset (deterministic, seed=42).

Outputs:
  data/raw/tickets.csv            noisy raw tickets (HTML, dupes, missing fields)
  data/raw/kb_articles.json       knowledge-base articles
  data/evaluation/eval_queries.jsonl  held-out complaints with ground-truth labels
"""
import csv
import json
import random
from datetime import date, timedelta
from pathlib import Path

SEED = 42
TICKETS_PER_SCENARIO = 80
EVAL_PER_SCENARIO = 6
NUM_TRAIN_SYMPTOMS = 3  # symptoms[3:] are held out for the eval set only

RAW_DIR = Path("data/raw")
EVAL_DIR = Path("data/evaluation")

# Each scenario: 5 symptom phrasings (first 3 for tickets, last 2 held out), 3 resolutions, 1 KB article.
SCENARIOS: dict[str, dict] = {
    "KB-001": dict(
        intent="connectivity_issue", product="broadband",
        symptoms=[
            "my broadband drops every evening around {t} PM",
            "internet disconnects repeatedly after {t} PM and the router light turns red",
            "connection becomes unstable during evening hours, error ERR-LOS-4 shows on the router",
            "wifi keeps cutting out at night even after I restarted the router twice",
            "the line goes down most evenings and comes back on its own later",
        ],
        resolutions=[
            "Ran a line stability test and found packet loss at peak hours. Moved the customer to a less congested port and the drops stopped.",
            "Replaced the faulty microfilter and reseated the DSL cable. Line sync stayed stable for 48 hours afterwards.",
            "Line statistics showed noise margin fluctuation. Raised a line-level fault and an engineer repaired the cabinet joint.",
        ],
        kb_title="Troubleshooting intermittent broadband disconnections",
        kb_body="Disconnections that cluster at the same time each day usually point to congestion or line noise. Step 1: power-cycle the router and check the DSL light. Step 2: run a line stability diagnostic and review packet loss. Step 3: check the microfilter and wiring. Step 4: if drops persist beyond 48 hours, raise a line-level fault for engineer investigation. Error ERR-LOS-4 indicates loss of signal.",
    ),
    "KB-002": dict(
        intent="slow_speed", product="broadband",
        symptoms=[
            "my download speed is only {t} Mbps but I pay for 100 Mbps",
            "internet is very slow, pages take forever to load and video calls lag",
            "speed test shows far below my plan speed on wifi",
            "everything crawls at night and streaming drops to low quality",
            "uploads are painfully slow although my plan promises fibre speeds",
        ],
        resolutions=[
            "Wired speed test was normal, so wifi interference was the cause. Moved the router to a 5 GHz channel and speeds returned to plan level.",
            "Found a misconfigured speed profile on the account and reset it to the 100 Mbps plan profile.",
            "Line test showed high attenuation. Engineer replaced the drop cable and speeds recovered.",
        ],
        kb_title="Diagnosing slow broadband speeds",
        kb_body="Slow speeds can come from wifi interference, wrong plan profile, or line attenuation. Step 1: run a speed test over a wired connection. Step 2: if wired is fine, switch wifi to the 5 GHz band. Step 3: verify the account speed profile matches the purchased plan. Step 4: run a line test and raise an engineer visit if attenuation is high.",
    ),
    "KB-003": dict(
        intent="billing_dispute", product="billing",
        symptoms=[
            "I was charged twice for my monthly bill this month",
            "there is an unexpected charge of ${t}9 on my invoice",
            "my bill is higher than the plan says and I do not recognise the add-on",
            "payment went through but the account still shows as overdue",
            "I cancelled a service last month but it still appears on my statement",
        ],
        resolutions=[
            "Verified a duplicate payment in the billing system and refunded the duplicate within 5 working days.",
            "Removed an add-on that was activated without consent and credited the amount to the account.",
            "Payment had been posted late. Cleared the overdue flag and waived the late fee.",
        ],
        kb_title="Handling billing disputes and duplicate charges",
        kb_body="Billing disputes must be verified against the payment ledger before any credit. Step 1: confirm the charge in the billing system. Step 2: check for duplicate payments or unconsented add-ons. Step 3: refund or credit within 5 working days. Step 4: clear overdue flags and waive late fees when the payment delay was ours.",
    ),
    "KB-004": dict(
        intent="network_coverage", product="mobile",
        symptoms=[
            "I have had no mobile signal inside my house for {t} days",
            "calls keep dropping in the middle of the conversation",
            "my phone shows emergency calls only",
            "signal bars disappear whenever I am indoors",
            "I cannot make or receive calls at my office anymore",
        ],
        resolutions=[
            "Nearby tower was under maintenance. Confirmed the restoration time and the signal returned the same day.",
            "Reset network settings and re-registered the device on the network. Calls became stable.",
            "Enabled Wi-Fi calling as a workaround for poor indoor coverage and logged the area for a coverage review.",
        ],
        kb_title="Resolving mobile coverage and call drop issues",
        kb_body="Coverage problems are either local (device or building) or network side (tower). Step 1: check the outage map for tower maintenance. Step 2: reset network settings and re-register the device. Step 3: enable Wi-Fi calling for poor indoor coverage. Step 4: log the location for a coverage review if it persists.",
    ),
    "KB-005": dict(
        intent="activation_issue", product="mobile",
        symptoms=[
            "my new SIM card is not activating and shows error SIM-ACT-12",
            "I inserted the replacement SIM but there is no network",
            "the activation SMS never arrived and the SIM is still inactive",
            "I ported my number but the new SIM stays on emergency calls only",
            "the SIM was delivered {t} days ago and still says not provisioned",
        ],
        resolutions=[
            "SIM was not provisioned in the activation system. Triggered manual provisioning and it went live within an hour.",
            "ICCID mismatch on the order. Corrected the SIM record and re-sent the activation.",
            "Port-in request was pending. Completed the port and the number activated.",
        ],
        kb_title="Fixing SIM activation failures",
        kb_body="Activation failures are usually provisioning or porting problems. Step 1: verify the ICCID on the order matches the SIM. Step 2: check the provisioning status and trigger manual provisioning. Step 3: for ported numbers, confirm the port-in request has completed. Error SIM-ACT-12 means the SIM is not provisioned.",
    ),
    "KB-006": dict(
        intent="streaming_issue", product="iptv",
        symptoms=[
            "TV channels keep buffering and error TV-503 appears on screen",
            "the streaming box freezes during live sports",
            "set-top box shows no signal on most channels",
            "on-demand movies stop and restart every few minutes",
            "the picture pixelates every evening on the IPTV service",
        ],
        resolutions=[
            "Set-top box firmware was outdated. Pushed the latest firmware and rebooted the box; buffering stopped.",
            "Home network was saturated. Prioritised IPTV traffic on the router and moved the box to a wired connection.",
            "Faulty HDMI cable and a weak coax joint. Replaced both and the picture became stable.",
        ],
        kb_title="Troubleshooting IPTV buffering and freezing",
        kb_body="Buffering is usually bandwidth, firmware, or cabling. Step 1: reboot the set-top box. Step 2: update the box firmware. Step 3: prefer a wired connection and prioritise IPTV traffic on the router. Step 4: replace faulty HDMI or coax cables. Error TV-503 means the stream server is unreachable from the box.",
    ),
    "KB-007": dict(
        intent="hardware_issue", product="router",
        symptoms=[
            "my router will not power on and shows no lights at all",
            "firmware update failed with error FW-209 and the router keeps rebooting",
            "the router is stuck in a boot loop after the update",
            "wifi name disappeared and the router lights are flashing orange",
            "the router overheats and shuts down randomly",
        ],
        resolutions=[
            "Power adapter was faulty. Shipped a replacement adapter and the router powered on.",
            "Recovered the router using the firmware recovery mode and reflashed the stable firmware version.",
            "Router hardware had overheating faults. Replaced the unit under warranty.",
        ],
        kb_title="Recovering routers after failed firmware updates",
        kb_body="A router stuck rebooting after a firmware update needs recovery, not repeated restarts. Step 1: check the power adapter and outlet. Step 2: boot into firmware recovery mode and reflash the stable firmware. Step 3: if the unit overheats or stays dead, arrange a warranty replacement. Error FW-209 means the firmware image failed verification.",
    ),
    "KB-008": dict(
        intent="roaming_issue", product="mobile",
        symptoms=[
            "no data while travelling abroad even though roaming is enabled",
            "I was charged heavy roaming fees on my trip",
            "I bought a roaming pack but my phone still has no internet abroad",
            "international roaming was not activated before my trip",
            "I landed {t} hours ago and only calls work, no mobile data",
        ],
        resolutions=[
            "Data roaming flag was off on the account. Enabled it and the phone connected to a partner network.",
            "Roaming pack had not been applied to the line. Applied it and credited the out-of-bundle charges.",
            "Manually selected the partner network and refreshed the APN settings; data worked.",
        ],
        kb_title="Fixing roaming data and charge problems",
        kb_body="Roaming issues come from account flags, packs, or network selection. Step 1: verify roaming is enabled on the account. Step 2: confirm the roaming pack is applied to the line. Step 3: select the partner network manually and refresh APN settings. Step 4: credit out-of-bundle charges caused by a missing pack.",
    ),
}

OPENERS = ["Hi,", "Hello support,", "Urgent:", ""]
DETAILS = ["It started last week.", "I have already restarted everything.", "This affects my whole family.", ""]
CLOSERS = {
    "neutral": ["Please advise.", "Let me know what to do."],
    "concerned": ["I work from home so this worries me.", "Please help soon."],
    "frustrated": ["This is the third time I am reporting it.", "I am really fed up."],
    "angry": ["This is unacceptable, I am considering cancelling.", "Fix this NOW or I am leaving."],
}
SENTIMENT_TO_SEVERITY = {
    "neutral": ["low", "medium"], "concerned": ["medium", "high"],
    "frustrated": ["high"], "angry": ["high", "critical"],
}
SEVERITY_ALIASES = {
    "low": ["low", "Low", "P4"], "medium": ["medium", "Med", "P3"],
    "high": ["high", "HIGH", "P2"], "critical": ["critical", "Critical", "P1"],
}


def build_complaint(rng: random.Random, symptom: str, sentiment: str) -> str:
    parts = [rng.choice(OPENERS), symptom.format(t=rng.randint(6, 10)).capitalize() + ".",
             rng.choice(DETAILS), rng.choice(CLOSERS[sentiment])]
    return " ".join(p for p in parts if p)


def add_noise(rng: random.Random, text: str) -> str:
    roll = rng.random()
    if roll < 0.10:
        return f"<p>{text}</p>"
    if roll < 0.20:
        return text.replace(" ", "  ") + "\n"
    return text


def main() -> None:
    rng = random.Random(SEED)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    eval_rows: list[dict] = []
    kb: list[dict] = []
    ticket_no = 10000

    for kb_id, sc in SCENARIOS.items():
        kb.append(dict(id=kb_id, title=sc["kb_title"], content=sc["kb_body"],
                       category=sc["intent"], product=sc["product"]))

        for _ in range(TICKETS_PER_SCENARIO):
            ticket_no += 1
            sentiment = rng.choice(list(CLOSERS))
            severity = rng.choice(SENTIMENT_TO_SEVERITY[sentiment])
            symptom = rng.choice(sc["symptoms"][:NUM_TRAIN_SYMPTOMS])
            complaint = build_complaint(rng, symptom, sentiment)
            rows.append(dict(
                ticket_id=f"T-{ticket_no}",
                created_at=(date(2025, 1, 1) + timedelta(days=rng.randint(0, 600))).isoformat(),
                subject=" ".join(symptom.split()[:7]).capitalize(),
                description=add_noise(rng, complaint),
                resolution_notes=rng.choice(sc["resolutions"]),
                category=sc["intent"].replace("_", " ").title(),
                product_line=sc["product"].upper(),
                priority=rng.choice(SEVERITY_ALIASES[severity]),
                customer_mood=sentiment.capitalize(),
                kb_ref=kb_id,
            ))

        for i in range(EVAL_PER_SCENARIO):
            sentiment = rng.choice(list(CLOSERS))
            severity = rng.choice(SENTIMENT_TO_SEVERITY[sentiment])
            symptom = rng.choice(sc["symptoms"][NUM_TRAIN_SYMPTOMS:])
            eval_rows.append(dict(
                query_id=f"Q-{kb_id}-{i + 1}",
                query=build_complaint(rng, symptom, sentiment),
                intent=sc["intent"], product=sc["product"],
                severity=severity, sentiment=sentiment,
                relevant_kb_id=kb_id,
            ))

    # Inject realistic mess so the cleaner has real work to do.
    for src in rng.sample(rows, 15):                        # duplicates
        ticket_no += 1
        rows.append({**src, "ticket_id": f"T-{ticket_no}"})
    for src in rng.sample(rows, 10):                        # unresolved tickets
        ticket_no += 1
        rows.append({**src, "ticket_id": f"T-{ticket_no}", "resolution_notes": "",
                     "description": src["description"] + " (still open)"})
    for _ in range(5):                                      # junk descriptions
        ticket_no += 1
        rows.append({**rows[0], "ticket_id": f"T-{ticket_no}", "description": "help"})
    rng.shuffle(rows)

    with (RAW_DIR / "tickets.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (RAW_DIR / "kb_articles.json").write_text(json.dumps(kb, indent=2), encoding="utf-8")
    with (EVAL_DIR / "eval_queries.jsonl").open("w", encoding="utf-8") as f:
        for r in eval_rows:
            f.write(json.dumps(r) + "\n")

    print(f"tickets={len(rows)} kb={len(kb)} eval_queries={len(eval_rows)}")


if __name__ == "__main__":
    main()