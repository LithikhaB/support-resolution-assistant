"""Validate evaluation packs and import explicitly supplied human ratings."""

import csv
import json
import re
from pathlib import Path

from app.ingestion.artifacts import file_sha256
from app.resolution.conversation import ConversationRequest

DIMENSIONS = ("context", "relevance", "conversation", "grounding", "clarity")


def validate_case_pack(pack, settings, *, holdout=False):
    """Reject empty, duplicated or training-overlapping cases before any model call."""
    if not isinstance(pack, dict) or any(
        not isinstance(pack.get(key), str) or not pack[key].strip()
        for key in ("provenance", "purpose")
    ):
        raise ValueError("Case pack requires purpose and provenance")
    cases = pack.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Case pack requires at least one case")
    ids, texts = set(), set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each case must be an object")
        identifier = case.get("id")
        if (
            not isinstance(identifier, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", identifier)
            or identifier in ids
        ):
            raise ValueError("Case IDs must be nonempty and unique")
        ids.add(identifier)
        request = ConversationRequest(query=case["query"], turns=case["turns"])
        if (
            not isinstance(case.get("expected"), list)
            or not case["expected"]
            or any(not isinstance(text, str) or not text.strip() for text in case["expected"])
        ):
            raise ValueError("Each case requires explicit review expectations")
        normalized = request.query.casefold().strip()
        if normalized in texts:
            raise ValueError("Duplicate evaluation complaint")
        texts.add(normalized)
    if holdout:
        known = set()
        for split in ("train", "dev", "test"):
            path = settings.corpus_dir / f"{split}.jsonl"
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    known.add(json.loads(line)["query"].casefold().strip())
        augmentation = settings.corpus_dir / "training_paraphrases.json"
        if augmentation.exists():
            families = json.loads(augmentation.read_text(encoding="utf-8"))["families"]
            known.update(
                text.casefold().strip() for queries in families.values() for text in queries
            )
        if texts & known:
            raise ValueError("Evaluation complaints overlap existing train/development/test data")
    return cases


def response_text(case):
    """Expose every conversation checkpoint so reviewers can judge answer evolution."""
    sections = []
    for snapshot in case["snapshots"]:
        sections.append(f"After reply {snapshot['after_reply']}")
        if "error" in snapshot:
            sections.append(f"Request error: {snapshot['error']}")
            continue
        for issue in snapshot["response"]["issues"]:
            result = issue["resolution"]
            plan = result.get("language_plan") or result["customer_plan"]
            sections.extend([plan["title"], result.get("language_summary") or plan["summary"]])
            sections.extend(plan["steps"])
            sections.append(plan.get("note", ""))
            sections.extend(result["clarification_questions"])
            for source in result["sources"]:
                sections.append(f"Source {source['doc_id']}:")
                sections.extend(f"{quote['field']}: {quote['text']}" for quote in source["quotes"])
    return "\n".join(sections)


def export_ratings(report, report_path: Path, output: Path):
    """Create a new review sheet with report identity and deliberately blank scores."""
    if report.get("run_complete") is not True or not report.get("results"):
        raise ValueError("Review requires a completed, nonempty report")
    checksum = file_sha256(report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "id",
        "report_sha256",
        "complaint",
        "replies",
        "responses",
        "review_expectations",
        *DIMENSIONS,
        "reviewer",
        "review_notes",
    )
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in report["results"]:
            row = {
                "id": case["id"],
                "report_sha256": checksum,
                "complaint": case["query"],
                "replies": "\n".join(turn["message"] for turn in case["turns"]),
                "responses": response_text(case),
                "review_expectations": "\n".join(case["expected"]),
            }
            writer.writerow(
                {
                    key: "'" + value
                    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@"))
                    else value
                    for key, value in row.items()
                }
            )


def import_ratings(report, report_path: Path, ratings: Path):
    """Accept only valid scores tied to this exact report, without mutating its evidence."""
    checksum = file_sha256(report_path)
    cases = {case["id"]: case for case in report["results"]}
    if len(cases) != len(report["results"]):
        raise ValueError("Duplicate case IDs in report")
    supplied = {}
    with ratings.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            identifier = row.get("id")
            if (
                identifier not in cases
                or identifier in supplied
                or row.get("report_sha256") != checksum
            ):
                raise ValueError("Ratings must have unique case IDs and match the exact report")
            scores = {}
            for dimension in DIMENSIONS:
                value = row.get(dimension)
                if value is None or value.strip() not in {"", "0", "1", "2"}:
                    raise ValueError("Scores must be 0, 1, 2 or blank")
                scores[dimension] = int(value) if value.strip() else None
            reviewer = row.get("reviewer", "").strip()
            if any(score is not None for score in scores.values()) and not reviewer:
                raise ValueError("Scored cases require a reviewer name or pseudonym")
            supplied[identifier] = {
                "manual_scores": scores,
                "reviewer": reviewer,
                "review_notes": row.get("review_notes", ""),
            }
    if supplied.keys() != cases.keys():
        raise ValueError("Ratings must include every case; leave unreviewed scores blank")
    return {
        **report,
        "results": [{**case, **supplied[case["id"]]} for case in report["results"]],
        "review_source_sha256": file_sha256(ratings),
        "unreviewed_report_sha256": checksum,
    }
