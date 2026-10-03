"""Small metrics with explicit denominators and no claims of independent variants."""

import numpy as np


def relevance(results, relevant_ids, k=5):
    """Measure a first relevant document within the first k ranked chunks."""
    identifiers = [r.doc_id for r in results[:k]]
    rank = next((i for i, doc_id in enumerate(identifiers, 1) if doc_id in relevant_ids), None)
    return {"hit": int(rank is not None), "reciprocal_rank": 1 / rank if rank else 0.0}


def routing_metrics(rows, scores, classes, min_score, min_margin, *, eligible=None):
    """Report acceptance coverage and conditional accuracy; empty acceptance has no accuracy."""
    ordered = np.sort(scores, axis=1)
    accepted = (ordered[:, -1] >= min_score) & (ordered[:, -1] - ordered[:, -2] >= min_margin)
    if eligible is not None:
        accepted &= np.asarray(eligible, dtype=bool)
    correct = np.asarray(
        [
            classes[int(np.argmax(score))] == row["labels"]["intent"]
            for row, score in zip(rows, scores, strict=True)
        ]
    )
    return {
        "accepted": int(accepted.sum()),
        "total": len(rows),
        "coverage": float(accepted.mean()),
        "accepted_accuracy": float(correct[accepted].mean()) if accepted.any() else None,
    }
