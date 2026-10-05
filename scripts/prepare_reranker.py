"""Download and warm the pinned local reranker before serving requests."""

from app.retrieval.reranking import RerankerUnavailable, get_reranking_service


def main():
    """Cache model assets and perform one local inference without touching the database."""
    try:
        service = get_reranking_service()
        for instance in getattr(service, "models", [service]):
            instance.model.predict(
                [("Broadband drops", "Intermittent broadband troubleshooting")],
                show_progress_bar=False,
            )
    except RerankerUnavailable as exc:
        raise SystemExit(
            "Reranker unavailable; check network access or the local model cache."
        ) from exc
    print(f"Ready: {service.settings.reranker_model}@{service.settings.reranker_revision}")


if __name__ == "__main__":
    main()
