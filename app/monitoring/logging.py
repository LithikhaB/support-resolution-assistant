import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure application logs without enabling raw complaint logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
