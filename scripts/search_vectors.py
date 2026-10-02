"""Run Phase E semantic retrieval without generating an answer."""
import argparse
import json

from app.retrieval.vector_search import VectorRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--queue")
    parser.add_argument("--doc-type", choices=["historical_response","resolved_ticket","knowledge_base"])
    args = parser.parse_args()
    filters = {k:v for k,v in {"queue":args.queue,"doc_type":args.doc_type}.items() if v is not None}
    results = VectorRetriever().search(args.query,args.top_k,filters)
    print(json.dumps({"results":[r.model_dump(mode="json") for r in results]},indent=2,ensure_ascii=False))


if __name__ == "__main__":
    main()
