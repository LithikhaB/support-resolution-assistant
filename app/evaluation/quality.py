"""Measure authored behavioural expectations and drift indicators separately from human ratings."""

from collections import Counter
from statistics import mean, median


def human_review_summary(cases):
    """Aggregate supplied 0–2 ratings without inventing missing human judgments."""
    dimensions = ("context", "relevance", "conversation", "grounding", "clarity")
    values = {name: [] for name in dimensions}
    completed = 0
    for case in cases:
        scores = case.get("manual_scores", {})
        required = set(dimensions) - ({"conversation"} if not case.get("turns") else set())
        for name in dimensions:
            score = scores.get(name)
            if score is not None:
                if type(score) is not int or not 0 <= score <= 2:
                    raise ValueError("Manual scores must be integer 0, 1, 2 or null")
                if name in required:
                    values[name].append(score)
        completed += all(scores.get(name) is not None for name in required)
    return {
        "fully_reviewed_cases": completed,
        "total_cases": len(cases),
        "dimensions": {
            name: {"rated_cases": len(scores), "mean_out_of_2": mean(scores) if scores else None}
            for name, scores in values.items()
        },
        "note": "Supplied manual ratings only; reviewer identity and independence are not verified by this script.",
    }


def check_snapshot(snapshot, checks):
    """Evaluate declarative facts, scope and questions without scoring writing quality."""
    if "error" in snapshot:
        return ["request_error"]
    issues = snapshot["response"]["issues"]
    failures = []
    if "issue_count" in checks and len(issues) != checks["issue_count"]:
        failures.append("issue_count")
    for expected in checks.get("issues", []):
        issue = next((item for item in issues if item["issue_id"] == expected["issue_id"]), None)
        if issue is None:
            failures.append("missing_issue")
            continue
        result, analysis = issue["resolution"], issue["resolution"]["analysis"]
        facts = {(fact["name"], fact["value"]) for fact in analysis["reported_facts"]}
        for name, value in expected.get("facts", {}).items():
            if (name, value) not in facts:
                failures.append(f"missing_fact:{name}")
        questions = " ".join(result["clarification_questions"]).lower()
        products = {product["product"] for product in analysis["products"]}
        for product in expected.get("products", []):
            if product not in products:
                failures.append("missing_product:" + product)
        for product in expected.get("excluded_products", []):
            if product in products:
                failures.append("incorrect_product:" + product)
        plan = result.get("language_plan") or result["customer_plan"]
        wording = " ".join(
            [result.get("language_draft") or result["draft"], plan["summary"], *plan["steps"]]
        ).lower()
        for text in expected.get("wording_excludes", []):
            if text.lower() in wording:
                failures.append("unsupported_wording:" + text)
        for text in expected.get("question_contains", []):
            if text.lower() not in questions:
                failures.append("missing_question:" + text)
        for text in expected.get("question_excludes", []):
            if text.lower() in questions:
                failures.append("repeated_question:" + text)
        if expected.get("no_questions") and result["clarification_questions"]:
            failures.append("unexpected_question")
        if expected.get("no_sources") and result["sources"]:
            failures.append("unexpected_sources")
        cited_scopes = {
            quote["text"]
            for source in result["sources"]
            for quote in source["quotes"]
            if quote["field"] == "scope"
        }
        if "allowed_source_scopes" in expected and not cited_scopes <= set(
            expected["allowed_source_scopes"]
        ):
            failures.append("unrelated_source_scope")
        if expected.get("requires_sources") and not result["sources"]:
            failures.append("missing_sources")
        for key in ("scope_status",):
            if key in expected and analysis[key] != expected[key]:
                failures.append(key)
        if (
            "allowed_categories" in expected
            and analysis["category"] not in expected["allowed_categories"]
        ):
            failures.append("incorrect_category")
        for key in ("action", "target", "priority"):
            if key in expected and result["decision"][key] != expected[key]:
                failures.append(key)
        if result["validation"]["status"] != "passed":
            failures.append("source_contract_failed")
    return failures


def summarize(report, golden):
    """Expose fallback and uncertainty rates so improvements cannot hide behind HTTP success."""
    expectations = {row["id"]: row for row in golden["cases"]}
    counters, providers, latencies, results = Counter(), Counter(), [], []
    for case in report["results"]:
        contract_failures = ["incomplete_run"] if report.get("run_complete") is False else []
        for snapshot in case["snapshots"]:
            counters["snapshots"] += 1
            if "error" in snapshot:
                counters["request_errors"] += 1
                contract_failures.append("request_error")
                continue
            for issue in snapshot["response"]["issues"]:
                response = issue["resolution"]
                if response["validation"]["status"] != "passed":
                    counters["source_contract_failures"] += 1
                    contract_failures.append("source_contract_failed")
                counters["issues"] += 1
                counters["uncertain_categories"] += response["analysis"]["category"] is None
                counters["needs_clarification"] += bool(response["clarification_questions"])
                counters["extractive_fallbacks"] += response.get("language_status") == "fallback"
                counters["language_drafts"] += (
                    response.get("language_status") == "generated_for_review"
                )
                counters["faithfulness_checked"] += (
                    response.get("faithfulness_status") == "model_checked"
                )
                providers[response.get("language_provider") or "extractive"] += 1
                latencies.append(response["elapsed_ms"])
        expected = expectations.get(case["id"])
        if expected:
            failures = contract_failures + (
                check_snapshot(case["snapshots"][-1], expected)
                if case["snapshots"]
                else ["missing_snapshots"]
            )
            for checkpoint in expected.get("checkpoints", []):
                snapshot = next(
                    (s for s in case["snapshots"] if s["after_reply"] == checkpoint["after_reply"]),
                    None,
                )
                failures.extend(
                    check_snapshot(snapshot, checkpoint) if snapshot else ["missing_checkpoint"]
                )
            failures = sorted(set(failures))
            results.append({"id": case["id"], "passed": not failures, "failures": failures})
    count = counters["issues"] or 1
    return {
        "run_complete": report.get("run_complete"),
        "counts": dict(counters),
        "providers": dict(providers),
        "rates": {
            key: counters[key] / count
            for key in ("uncertain_categories", "needs_clarification", "extractive_fallbacks")
        },
        "median_issue_ms": median(latencies) if latencies else None,
        "golden_passed": sum(row["passed"] for row in results),
        "golden_checked": len(results),
        "golden_results": results,
        "human_review": human_review_summary(report["results"]),
        "scope": "Authored development behaviour checks; not independent human answer-quality ratings or a production drift significance test.",
    }
