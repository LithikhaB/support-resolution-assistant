"""Build a source-based demo audit and an unrated human review sheet from a saved run."""

import argparse
import html
import json
from pathlib import Path

EXPECTED = {
    "evening": {"syn_kb_ID02", "syn_kb_ID04"},
    "storm": {"syn_kb_ID01", "syn_kb_ID04"},
    "optical": {"syn_kb_BB01"},
}
RUBRIC = {
    "relevance": "Every procedure is relevant to the reported symptoms.",
    "grounding": "Checks and remedies follow the quoted sources, without added claims.",
    "conditionality": "The cause remains unconfirmed and repairs require the full diagnostic gate.",
    "prior_actions": "Completed checks are acknowledged and are not recommended again.",
    "readability": "An agent can follow the plan without interpreting engineering shorthand.",
}


def audit_record(record):
    """Check selected sources and actual plan content, beyond HTTP/citation contracts."""
    response = record["local"]
    plan = response["customer_plan"]
    text = " ".join(plan["steps"])
    sources = {s["doc_id"] for s in response["sources"]}
    quotes = {
        s["citation_id"]: {q["field"]: q["text"] for q in s["quotes"]} for s in response["sources"]
    }
    return {
        "expected_demo_sources": sources == EXPECTED[record["case"]],
        "full_gates_present": all(s["required_finding"] in text for s in response["suggestions"]),
        "remedies_conditional": all(
            any(
                step.startswith("Only if support confirms") and s["proposed_action"] in step
                for step in plan["steps"]
            )
            for s in response["suggestions"]
            if not s["repeated_actions"]
        ),
        "completion_present": all(q["completion"] in text for q in quotes.values()),
        "prior_actions_acknowledged": all(
            a in plan["steps"][0] for a in response["acknowledged_actions"]
        ),
        "restrictions_preserved": all(
            s["restriction"] in plan["note"] for s in response["suggestions"]
        ),
        "step_count_proxy": len(plan["steps"]) <= 7,
    }


def build(report_path, output):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    audits = {r["case"]: audit_record(r) for r in report["records"]}
    audit_path = report_path.with_name(report_path.stem + "_content_audit.json")
    with audit_path.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "scope": "Three authored demo cases; semantic regression checks, not independent human ratings or held-out accuracy.",
                "independent_human_ratings": 0,
                "source_report": str(report_path),
                "checks": audits,
            },
            stream,
            indent=2,
        )
    esc = html.escape
    sections = []
    for record in report["records"]:
        response = record["local"]
        plan = response["customer_plan"]
        evidence = ""
        for source in response["sources"]:
            evidence += (
                f"<details><summary>[{esc(source['citation_id'])}] {esc(source['title'])}</summary>"
            )
            evidence += "".join(
                f"<p><strong>{esc(q['field'])}:</strong> {esc(q['text'])}</p>"
                for q in source["quotes"]
            )
            evidence += "</details>"
        ratings = "".join(
            f'<label>{esc(description)} <select data-case="{esc(record["case"])}" data-area="{area}"><option value="">Unrated</option><option>pass</option><option>fail</option></select></label>'
            for area, description in RUBRIC.items()
        )
        steps = "".join(f"<li>{esc(step)}</li>" for step in plan["steps"])
        sections.append(
            f'<article><h2>{esc(record["case"].title())}</h2><blockquote>{esc(record["query"])}</blockquote><p><b>Decision:</b> {esc(response["decision"]["action"])} · {esc(response["decision"]["priority"])}</p><p>Local deterministic plan · AI-authored synthetic sources · no confirmed diagnosis</p><ol>{steps}</ol><p>{esc(plan["note"])}</p><p><b>Pending question:</b> {esc(" ".join(response["clarification_questions"]))}</p>{evidence}<h3>Your quality rating</h3>{ratings}<label>Corrections or concerns<textarea data-notes="{esc(record["case"])}" rows="3"></textarea></label></article>'
        )
    document = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Review demo plan quality</title><style>body{font:16px/1.6 system-ui;background:#f4f7f6;color:#18332f;max-width:850px;margin:30px auto;padding:16px}article{background:white;border:1px solid #bdd2c9;border-radius:12px;padding:24px;margin:24px 0}li{margin:12px 0}blockquote{margin:0;padding:16px;background:#f1f9f5}details{padding:10px;border-bottom:1px solid #d8e3de}label{display:block;margin:12px 0}select{margin-left:12px;padding:6px}textarea{display:block;width:95%;padding:10px;font:inherit}button{padding:12px 18px;background:#087b66;color:white;border:0;border-radius:8px;font:inherit}summary{cursor:pointer}#status{font-weight:bold}</style><h1>Review demo plan quality</h1><p>This sheet contains the three actual local API responses. Automated content checks pass, but human ratings are currently <b>unrated</b>. They are separate from citation validation and model critique. Source correctness in real telecom operations still needs expert review.</p>"""
    document += "".join(sections)
    document += """<label><input id="confirmed" type="checkbox"> I personally reviewed all three plans and their source evidence.</label><button id="save">Download my ratings</button><p id="status" role="status"></p><script>document.getElementById('save').onclick=()=>{const ratings=[...document.querySelectorAll('select')];if(!document.getElementById('confirmed').checked||ratings.some(r=>!r.value)){document.getElementById('status').textContent='Review every area and confirm your personal review before exporting.';return;}const result={reviewer_type:'human_self_attested',reviewed_at:new Date().toISOString(),ratings:ratings.map(r=>({case:r.dataset.case,area:r.dataset.area,rating:r.value})),notes:[...document.querySelectorAll('textarea')].map(n=>({case:n.dataset.notes,notes:n.value}))};const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));const link=document.createElement('a');link.href=url;link.download='human-plan-quality-ratings.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);document.getElementById('status').textContent='Ratings exported locally. No provider request or account action was made.';};</script></html>"""
    output.write_text(document, encoding="utf-8")
    print(
        json.dumps(
            {
                "content_checks": audits,
                "human_review_sheet": str(output),
                "audit_report": str(audit_path),
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("app/web/plan-review.html"))
    args = parser.parse_args()
    build(args.report, args.output)


if __name__ == "__main__":
    main()
