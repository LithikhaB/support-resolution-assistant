"""Select abstention thresholds on development families without changing accuracy floors."""

from collections import defaultdict

import numpy as np

from app.understanding.routing import compatible_category
from app.understanding.signals import extract_products


def select_thresholds(
    rows,
    scores,
    classes,
    *,
    target_accuracy=0.85,
    minimum_families=5,
    minimum_examples=20,
    category_products=None,
):
    """Maximize accepted coverage subject to example and family-average accuracy floors."""
    labels = np.asarray([r["labels"]["intent"] for r in rows])
    order = np.argsort(-scores, axis=1, kind="stable")
    best = scores[np.arange(len(scores)), order[:, 0]]
    margins = best - scores[np.arange(len(scores)), order[:, 1]]
    correct = np.asarray([classes[i] for i in order[:, 0]]) == labels
    compatible = np.asarray(
        [
            compatible_category(classes[index], extract_products(row["query"]), category_products)
            for row, index in zip(rows, order[:, 0], strict=True)
        ]
    )
    choices = []
    for threshold in np.arange(0.10, 0.51, 0.01):
        for margin in np.arange(0.00, 0.21, 0.01):
            selected = (best >= threshold) & (margins >= margin) & compatible
            families = defaultdict(list)
            for row, accepted, hit in zip(rows, selected, correct, strict=True):
                if accepted:
                    families[row["scenario_family"]].append(bool(hit))
            if selected.sum() < minimum_examples or len(families) < minimum_families:
                continue
            accuracy = float(correct[selected].mean())
            family_accuracy = float(np.mean([np.mean(values) for values in families.values()]))
            if min(accuracy, family_accuracy) < target_accuracy:
                continue
            choices.append(
                {
                    "min_score": round(float(threshold), 2),
                    "min_margin": round(float(margin), 2),
                    "accepted": int(selected.sum()),
                    "accepted_families": len(families),
                    "accepted_accuracy": accuracy,
                    "family_mean_accuracy": family_accuracy,
                    "total": len(rows),
                }
            )
    return (
        max(
            choices,
            key=lambda c: (
                c["accepted"],
                c["family_mean_accuracy"],
                c["min_margin"],
                c["min_score"],
            ),
        )
        if choices
        else None
    )
