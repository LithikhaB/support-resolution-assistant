"""Stable evidence selection that limits repeated documents and synthetic variants."""


def diverse_results(results, limit):
    """Keep one chunk per document and one synthetic case per source family and type."""
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    selected = []
    seen = set()
    for result in results:
        metadata = result.metadata
        family = metadata.get("scenario_family")
        if metadata.get("is_synthetic") is True and isinstance(family, str) and family.strip():
            key = (
                "family",
                result.doc_type,
                str(metadata.get("source", "")),
                str(metadata.get("dataset_version", "")),
                family,
            )
        else:
            key = ("document", result.doc_id)
        if key in seen:
            continue
        seen.add(key)
        selected.append(result)
        if len(selected) == limit:
            break
    return selected
