"""Validate optional training paraphrases without adding evaluation families or labels."""

import json

from app.ingestion.artifacts import file_sha256


def augment_training(rows, path):
    """Attach paraphrases only to existing training families and retain their intent label."""
    if not path.exists():
        return rows, None
    checksum = file_sha256(path)
    pack = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(pack, dict) or not isinstance(pack.get("provenance"), str):
        raise ValueError("Training paraphrases require provenance")
    families = pack.get("families")
    if not isinstance(families, dict) or not families:
        raise ValueError("Training paraphrases require a nonempty family mapping")
    originals = {}
    for row in rows:
        family = row["scenario_family"]
        if family in originals and originals[family]["labels"]["intent"] != row["labels"]["intent"]:
            raise ValueError("Training family has conflicting intent labels")
        originals[family] = row
    seen = {row["query"].strip().casefold() for row in rows}
    ids = {row["query_id"] for row in rows}
    augmented = list(rows)
    for family, queries in families.items():
        if family not in originals or not isinstance(queries, list) or not queries:
            raise ValueError("Paraphrases must belong to existing training families")
        for index, query in enumerate(queries, 1):
            if not isinstance(query, str) or not query.strip() or len(query) > 10000:
                raise ValueError("Invalid training paraphrase")
            normalized = query.strip().casefold()
            query_id = f"paraphrase_{family}_{index}"
            if normalized in seen or query_id in ids:
                raise ValueError("Duplicate training paraphrase or query ID")
            seen.add(normalized)
            ids.add(query_id)
            augmented.append(
                {
                    "query_id": query_id,
                    "query": query.strip(),
                    "split": "train",
                    "scenario_family": family,
                    "labels": {"intent": originals[family]["labels"]["intent"]},
                }
            )
    if file_sha256(path) != checksum:
        raise ValueError("Training paraphrases changed while loading")
    return augmented, checksum
