"""Inspect local category predictions, impact, tone and troubleshooting observations."""

import argparse

from pydantic import ValidationError

from app.retrieval.embeddings import EmbeddingInputTooLong
from app.understanding.classifier import UnderstandingUnavailable
from app.understanding.models import AnalyzeRequest
from app.understanding.service import get_understanding_service


def main() -> None:
    """Print validated analysis or a concise actionable error."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    args = parser.parse_args()
    try:
        request = AnalyzeRequest(query=args.query)
        result = get_understanding_service().analyze(request)
    except ValidationError:
        parser.exit(2, "Provide a non-empty complaint of at most 10000 characters.\n")
    except EmbeddingInputTooLong:
        parser.exit(2, "Complaint exceeds the embedding token budget; shorten it.\n")
    except (UnderstandingUnavailable, OSError):
        parser.exit(
            1,
            "Understanding unavailable; run scripts.train_understanding and check the model cache.\n",
        )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
