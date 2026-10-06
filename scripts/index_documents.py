"""Index verified local artifacts in restartable batches; never reset the database."""

import argparse
import json

from app.config.settings import get_settings
from app.ingestion.indexing import index_corpus
from app.monitoring.logging import configure_logging


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Re-embed tracked documents when the runtime configuration changes; preserve source rows and resume completed batches.",
    )
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    print(json.dumps(index_corpus(settings, rebuild=args.rebuild), indent=2))


if __name__ == "__main__":
    main()
