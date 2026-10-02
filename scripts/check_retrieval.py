"""Run manual smoke queries. This is NOT a labeled relevance benchmark."""
import argparse
import json
from pathlib import Path

from app.retrieval.models import SearchRequest
from app.retrieval.service import get_retrieval_service


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries', type=Path, default=Path('data/evaluation/retrieval_smoke_queries.jsonl'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    service = get_retrieval_service()
    runs = []
    for line in args.queries.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        for mode in ('bm25', 'vector', 'hybrid'):
            result = service.search(SearchRequest(query=case['query'], mode=mode, top_k=3,
                                                  filters=case.get('filters', {})))
            runs.append({'query_id': case['id'], 'review_note': case['review_note'],
                         **result.model_dump(mode='json')})
    report = {'purpose': 'Manual smoke check only; no relevance labels or accuracy metrics.',
              'timing_note': 'Sequential local runs; initial requests include cache/model loading.',
              'runs': runs}
    encoded = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + '\n', encoding='utf-8')
        print(f'Wrote {len(runs)} smoke runs to {args.output}')
    else:
        print(encoded)


if __name__ == '__main__':
    main()
