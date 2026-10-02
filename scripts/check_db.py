"""Check the live Day 1 database without changing any data."""
from app.api.routes import readiness


def main() -> None:
    result = readiness()
    if isinstance(result, dict):
        print("Database ready: pgvector, documents, chunks and Day 1 schema found.")
    else:
        print(result.body.decode("utf-8"))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
