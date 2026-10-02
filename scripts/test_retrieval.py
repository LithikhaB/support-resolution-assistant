"""Readable Day 2 retrieval demo. Use --json for machine-readable evidence."""
import argparse
import json

import psycopg
from pydantic import ValidationError

from app.retrieval.embeddings import EmbeddingInputTooLong
from app.retrieval.models import SearchRequest
from app.retrieval.service import get_retrieval_service
from app.retrieval.vector_search import RetrievalUnavailable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('query')
    parser.add_argument('--mode', choices=['hybrid', 'bm25', 'vector'], default='hybrid')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--candidate-k', type=int)
    parser.add_argument('--queue')
    parser.add_argument('--doc-type', choices=['historical_response', 'resolved_ticket', 'knowledge_base'])
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    try:
        filters = {k: v for k, v in {'queue': args.queue, 'doc_type': args.doc_type}.items() if v is not None}
        request = SearchRequest(query=args.query, mode=args.mode, top_k=args.top_k,
                                candidate_k=args.candidate_k, filters=filters)
        response = get_retrieval_service().search(request)
    except ValidationError as exc:
        parser.error('; '.join(e['msg'] for e in exc.errors()))
    except EmbeddingInputTooLong:
        parser.exit(2, 'Query exceeds the embedding token limit; shorten the complaint.\n')
    except psycopg.errors.QueryCanceled:
        parser.exit(1, 'Retrieval timed out; retry or narrow the filters.\n')
    except (RetrievalUnavailable, psycopg.Error, OSError):
        parser.exit(1, 'Retrieval unavailable; check database, index and local model cache.\n')
    if args.json:
        print(response.model_dump_json(indent=2))
        return
    print(f'QUERY: {request.query}\n{response.mode.upper()} RESULTS ({response.elapsed_ms:.1f} ms)')
    if not response.results:
        print('No matching evidence found.')
    for rank, result in enumerate(response.results, 1):
        details = result.model_dump(mode='json')
        scores = {k: v for k, v in details.items() if k.endswith(('_rank', '_score', '_contribution'))
                  or k == 'cosine_similarity'}
        print(f'\nRank {rank} | Document: {result.doc_id} | Chunk: {result.chunk_id}')
        print(f'Title: {result.title}\nSources: {", ".join(result.sources)}')
        print('Scores/ranks: ' + json.dumps(scores))
        print(f'Content: {result.content}\nHistorical response: {result.response}')
        print(f'Outcome: {result.outcome_status} | Verified resolution: {result.resolution}')


if __name__ == '__main__':
    main()
