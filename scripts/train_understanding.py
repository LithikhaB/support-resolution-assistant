"""Fit two local category baselines and select using the development split only."""

import json

from app.config.settings import get_settings
from app.monitoring.logging import configure_logging
from app.understanding.training import train_classifier


def main() -> None:
    """Publish the selected classifier and report actual development measurements."""
    settings = get_settings()
    configure_logging(settings.log_level)
    report = train_classifier(settings)
    print(
        json.dumps(
            {
                "selected": report["selected"],
                "test_split_used": report["test_split_used"],
                "model_path": str(settings.understanding_model_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
