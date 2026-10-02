# Day 2: retrieval implementation and verification

Day 2 retrieval code is complete through phases H and I. The README is intentionally deferred at the user's request. This service returns evidence; query understanding, reranking, generated answers and the UI belong to later days.

## What is available

- `app/retrieval/service.py`: one application-facing service for BM25, vector and hybrid modes. The API reuses it within each process.
- `POST /api/v1/retrieve`: validated inputs, ranked evidence, source ranks/contributions and elapsed time.
- `python -m scripts.test_retrieval`: readable CLI demo; `--json` returns structured output.
- `python -m scripts.check_retrieval`: repeatable manual smoke checks across all three modes.
- Retrieval logs include mode, query length, result count and timings. Hybrid logs separate BM25, vector and fusion time. Complaint text is not logged by these components.

## Run locally

Use the repository root with the virtual environment active and PostgreSQL running. Existing indexed data can be reused; no reindex or new dependency installation is required for H/I.

```powershell
python -m scripts.check_index
python -m scripts.test_retrieval "Connection keeps dropping" --queue technical_support --top-k 5
python -m scripts.test_retrieval "Salesforce Microsoft Teams integration" --mode bm25 --top-k 3
python -m uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs and use **POST /api/v1/retrieve**, or run this in another PowerShell terminal:

```powershell
$body = @{
    query = "Connection keeps dropping"
    top_k = 5
    mode = "hybrid"
    filters = @{ queue = "technical_support" }
} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/v1/retrieve" -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 12
```

The response contains `mode`, `results` and `elapsed_ms`. Each result carries complaint text, parent evidence, IDs and score provenance. Historical replies retain `outcome_status=unknown` and `resolution=null`.

Requests accept `mode` as `bm25`, `vector` or `hybrid` (default), `top_k` from 1 to 100 (default 5), and the supported filters: queue, doc_type, intent, severity and ticket_type. Optional `candidate_k` applies only to hybrid, must be at least top_k, and is bounded at 100.

## Configuration and expected behavior

- `RETRIEVAL_CANDIDATE_K=50`: candidate depth per branch. If omitted from a request, the effective depth is at least top_k.
- `RETRIEVAL_RRF_CONSTANT=60`: each participating branch contributes `1 / (constant + rank)`. RRF scores are not confidence probabilities.
- `RETRIEVAL_STATEMENT_TIMEOUT_MS=30000`: database statement timeout, separate from the short health-check timeout. It does not bound model loading or total request time.
- The first request loads the in-memory BM25 index and, for semantic modes, the local embedding model. Later requests in the same process reuse them. Reloads and additional workers have separate caches.
- With a fully cached model, `EMBEDDING_LOCAL_FILES_ONLY=true` prevents model downloads. If files are missing, populate the pinned cache first.
- Vector/hybrid queries exceeding the model's 256-token ceiling are rejected, not silently truncated. This token bound is separate from the API's 10,000-character bound.
- Filtered vector queries perform exact ranking over eligible chunks; unfiltered queries use approximate HNSW search.
- BM25 automatically rebuilds its process cache when indexing-state fingerprints/completion time change. Indexing and retrieval coordinate through database advisory locks.

## Errors

| Status | Meaning | Action |
|---|---|---|
| 200 with empty results | No eligible evidence, or no lexical matches in BM25 mode | Check query/filter values |
| 422 | Invalid request or embedding token budget exceeded | Correct fields or shorten the complaint |
| 503 | Database/index/model unavailable or indexing in progress | Check PostgreSQL, `scripts.check_index`, model cache; retry after indexing |
| 504 | Database statement timed out | Retry; inspect query plan/load and retrieval timeout if persistent |

The existing `/api/v1/ready` checks Day 1 database/schema readiness only. It does not assert that the retrieval index or model is ready. Use `scripts.check_index` and a retrieval request to check those.

## Verification recorded on 2026-10-02

- Full suite with `RUN_DB_TESTS=1`: **118 passed**, including PostgreSQL tests in temporary schemas whose changes are rolled back. Dependency deprecation warnings remain in FastAPI/Starlette.
- Read-only index check: **23,790 documents; 23,811 chunks and embeddings; zero invalid embeddings; valid cosine HNSW index**.
- Real FastAPI TestClient request against the corpus and local model: HTTP 200 with five technical-support results. An unmatched queue returned an empty list. Health and OpenAPI returned HTTP 200.
- Real CLI keyword search found a Salesforce/Microsoft Teams integration ticket and preserved its unknown outcome.
- Nine smoke runs (three themes × three modes) each returned three candidates. Full evidence is in `data/evaluation/day2_smoke_report.json`; query definitions are in `data/evaluation/retrieval_smoke_queries.jsonl`.

These are functional checks, not a labeled evaluation. No Recall@K, MRR or accuracy claim is made. Inspection found an off-topic BM25 top result for “Connection keeps dropping”; vector search returned network-disconnection tickets, while hybrid favored repeated smart-device connectivity examples. Reranking and duplicate-aware evidence selection remain future work. The smoke report retains these outcomes rather than selecting only favorable examples.

Initial loading took several seconds for BM25 and tens of seconds for the semantic model in this session. Subsequent smoke requests were faster, but these sequential local measurements are not a controlled performance benchmark.

Reproduce:

```powershell
# Ordinary tests: no running database or model download required
python -m pytest tests/unit -q -p no:cacheprovider

# Includes PostgreSQL integration tests
$env:RUN_DB_TESTS="1"
python -m pytest -q -p no:cacheprovider

# Uses the real indexed corpus and cached model
python -m scripts.check_retrieval --output data/evaluation/day2_smoke_report.json
```

Manually inspect the complaint, source ranks, contributions and historical reply together. A retrieved reply is evidence of a past response, not proof that it fixed the customer's problem.
