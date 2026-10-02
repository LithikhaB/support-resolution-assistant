"""Search the corpus using BM25, vector search, or reciprocal rank fusion."""
import argparse
import json

from app.retrieval.bm25_search import BM25Retriever
from app.retrieval.hybrid_search import HybridRetriever
from app.retrieval.vector_search import VectorRetriever


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('query')
    parser.add_argument('--mode', choices=['bm25', 'vector', 'hybrid'], default='hybrid')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--candidate-k', type=int, default=50)
    parser.add_argument('--queue')
    parser.add_argument('--doc-type', choices=['historical_response','resolved_ticket','knowledge_base'])
    args = parser.parse_args()
    filters = {k: v for k, v in {'queue': args.queue, 'doc_type': args.doc_type}.items() if v is not None}
    retriever = {'bm25': BM25Retriever, 'vector': VectorRetriever, 'hybrid': HybridRetriever}[args.mode]()
    kwargs = {'candidate_k': args.candidate_k} if args.mode == 'hybrid' else {}
    results = retriever.search(args.query, args.top_k, filters, **kwargs)
    print(json.dumps({'mode': args.mode, 'results': [r.model_dump(mode='json') for r in results]},
                     indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
