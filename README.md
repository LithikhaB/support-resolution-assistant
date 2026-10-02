# AI-Powered Telecom Support Resolution Assistant

A local-first support retrieval service for the Synapt hiring challenge.

## Problem and understanding

An agent receives a complaint such as: “My broadband drops every evening. I already restarted the router twice, and I work from home.” Keyword search can miss equivalent symptoms, and a similar ticket does not necessarily establish the same cause.

The complete application should:

1. Identify the service, category, impact, sentiment and attempted troubleshooting.
2. Retrieve relevant historical cases and knowledge-base procedures.
3. Draft evidence-grounded next steps with citations.
4. Ask for missing information or escalate when a diagnosis is unsupported.
5. Accept new knowledge and ticket categories without redesigning storage.

## Current implementation

- Deterministic synthetic telecom dataset with linked diagnostic findings and KB procedures.
- Validated evidence schema distinguishing simulated outcomes from real verification.
- Token-aware chunks and pinned local CPU embeddings.
- PostgreSQL/pgvector indexing with transactional batches, safe resume and cosine HNSW.
- BM25 keyword search, vector search and Reciprocal Rank Fusion.
- FastAPI retrieval and complaint-analysis endpoints, local CLIs and regression tests.
- Trained local category classifier plus explainable product, impact, sentiment, prior-action and reported-condition rules.
- Optional local cross-encoder reranking over a bounded retrieval pool.

Answer generation, citation-support validation and a UI are **not implemented yet**. Current responses are ranked evidence, not generated resolutions. Retrieval does not guarantee relevance or automatically abstain on out-of-domain queries.

## Architecture

![Project architecture](docs/architecture%20v1.svg)

The SVG shows the full project direction. The implemented request path is:

`API / CLI → RetrievalService → BM25 + pgvector → RRF → ranked evidence`

## Dataset

The primary corpus is `data/synthetic/telecom_v1`, authored for a fictional provider. The previous Hugging Face corpus was useful for engineering retrieval, but its broad domains and unverified replies do not supply focused telecom resolution evidence.

| Artifact | Count and purpose |
|---|---|
| Scenario families | 60 distinct authored diagnostic situations across 15 categories |
| Tickets | 480 simulated resolved variants and 30 unresolved cases |
| KB | 60 conditional fictional-provider procedures |
| Train / development / test | 240 / 120 / 120 labeled queries; families never cross splits |
| Challenge cases | 15 ambiguity, safety, unsupported-request and repeated-action cases |
| Indexed corpus | 330 documents: 240 training tickets, 30 unresolved cases and 60 KB articles |

Categories: broadband outage, intermittent broadband, slow broadband, Wi-Fi connectivity, router/ONT hardware, mobile coverage, voice failure, mobile data, SIM/eSIM activation, porting, SMS/OTP, roaming, billing disputes, payment restoration and IPTV.

Only `processed/documents.jsonl` is indexed. The full `tickets.jsonl` includes held-out examples and must not be used wholesale as a retrieval corpus. Each scenario has controlled wording/tone variants; those variants are not independent real incidents. Outcome status is `simulated_resolved` or `unknown`, never fabricated real verification.

The corpus has automatic integrity checks, not expert certification. Human review and independent evaluation questions remain necessary. All KBs are available at evaluation time; this tests unseen tickets against existing knowledge, not unseen knowledge. See [interview notes](docs/interview-notes.md).

## Setup and run

Use Python 3.12 or newer, PostgreSQL 16 with pgvector, and a CPU-capable machine. Commands below are PowerShell from the repository root.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Set database credentials in `.env`. The active defaults are:

```dotenv
POSTGRES_DB=support_telecom
CORPUS_DIR=data/synthetic/telecom_v1
```

Start PostgreSQL if needed. Skip Docker startup if your existing PostgreSQL server is already running on the configured port.

```powershell
docker compose up -d
python -m scripts.setup_database
python -m scripts.prepare_synthetic
python -m scripts.chunk_documents
python -m scripts.index_documents
python -m scripts.check_index
```

`setup_database` creates the configured database if missing, then initializes tables and constraints. The database user needs create-database and extension privileges, or an administrator must provision them first. Existing rows are preserved. Do not use `docker compose down -v` to switch datasets.

First use may download the pinned model into `data/models`. After caching, offline operation can be enabled:

```powershell
$env:EMBEDDING_LOCAL_FILES_ONLY="true"
$env:TOKENIZER_LOCAL_FILES_ONLY="true"
```

Expected index check: `status=ready`, 330 documents, 330 embeddings, zero invalid embeddings and a valid cosine HNSW index. Reindexing unchanged artifacts skips existing documents. Old Hugging Face rows remain in the separate `support_db` database, not the active index.

### CLI

```powershell
python -m scripts.test_retrieval "Wi-Fi is poor upstairs but Ethernet works" --top-k 5
python -m scripts.test_retrieval "Two payments for one invoice have both settled" --mode bm25 --queue billing_support
python -m scripts.test_retrieval "Outgoing calls fail but mobile data works" --mode vector --product mobile_voice --json
```

Modes are `hybrid` (default), `vector`, and `bm25`. Filters include queue, intent, product and document type. Use `--help` for arguments. Historical response and simulated resolution are displayed separately.

### Local complaint understanding

Train or rebuild the classifier, then analyze a complaint:

```powershell
python -m scripts.train_understanding
python -m scripts.analyze_complaint "My broadband drops every evening. I already restarted the router twice."
```

Training compares TF-IDF and frozen MiniLM features with logistic regression at three regularization settings. Only train/development splits are loaded. The selected JSON weights are saved under `data/models/understanding`; restart the API after retraining.

Current development result: **60% accuracy, 0.552 macro-F1**, across 120 synthetic variants from 15 families. This is model-selection evidence, not a test-set or real-world result. Detailed comparisons are in `data/evaluation/understanding_development.json`.

### API

```powershell
python -m uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs. Submit to `POST /api/v1/retrieve`:

```json
{
  "query": "Wi-Fi is poor upstairs but Ethernet works",
  "top_k": 5,
  "mode": "hybrid",
  "filters": {"intent": "wifi_connectivity"}
}
```

For complaint understanding, submit to `POST /api/v1/analyze`:

```json
{"query": "My broadband drops every evening. I already restarted the router twice."}
```

Analysis returns category candidates, explicitly mentioned products, severity, sentiment, attempted/negated/suggested actions, reported facts, contact/next-step requests, supporting text spans and clarification questions. A reported intact cable remains a customer statement. Contact requests ask for provider and region; the application does not invent a phone number. Known service details guide follow-up questions. It runs locally without database access. Low category scores or small score margins return `category_status=uncertain`; scores are uncalibrated and do not reliably detect unsupported topics. English rules can miss unfamiliar phrasing. Analysis and retrieval remain separate; predicted categories do not automatically filter evidence.

The retrieval response includes evidence, document IDs, source ranks, score contributions and elapsed milliseconds. `GET /api/v1/health` checks process liveness; `GET /api/v1/ready` checks database schema availability only. `scripts.check_index` checks the retrieval index.

| Response | Meaning |
|---|---|
| 200, empty results | No eligible documents or no lexical matches |
| 422 | Invalid fields, candidate depth, or model token budget exceeded |
| 503 | Database/index/model unavailable or indexing in progress |
| 504 | Database statement timed out |

Queries are limited to 10,000 characters; semantic modes also enforce the model's 256-token ceiling. `top_k` is 1–100. Hybrid `candidate_k` must be at least top_k and at most 100. Unsupported filter fields are rejected. There is no automatic query truncation.

### Optional local reranking

Prepare the pretrained reranker once, then compare rankings:

```powershell
python -m scripts.prepare_reranker
python -m scripts.test_retrieval "My broadband drops every evening. I restarted the router twice." --rerank --rerank-k 20 --top-k 5
python -m scripts.evaluate_reranking
```

For Swagger, use `POST /api/v1/retrieve`:

```json
{
  "query": "My broadband drops every evening. I already restarted the router twice.",
  "mode": "hybrid",
  "top_k": 5,
  "rerank": true,
  "rerank_k": 20
}
```

`rerank_k` is the pool scored by the local cross-encoder (maximum 50, at least `top_k`). It defaults to the larger of 20 and `top_k`. Hybrid `candidate_k`, when supplied, must cover this pool. All three retrieval modes support reranking, using the caller's filters. Empty pools return no results without loading the reranker. Missing models return 503 rather than silently changing the requested ranking method.

Results retain original ranks and evidence alongside `rerank_score` and `rerank_rank`. Scores measure relevance, not correctness or resolution confidence. Query inputs are limited to 256 reranker tokens; pairs exceeding 512 tokens are truncated for scoring and flagged with `rerank_input_truncated`, while returned evidence stays complete. The first request loads models; later requests reuse them. After preparation, set `RERANKER_LOCAL_FILES_ONLY=true` for offline loading.

The development comparison uses the same 20 candidates for each ranking method and provisional authored KB relevance labels. It does not evaluate final answers or real customer outcomes. Across 120 development queries, expected-KB hit@5 was **50.0% for hybrid versus 36.7% with reranking**; MRR@5 was 0.461 versus 0.341. The expected KB was present in only 62.5% of the 20-candidate pools. Median reranking time was about 759 ms on this machine. This measured regression is why reranking remains optional. Repeated scenario variants and candidate coverage need further work. See `data/evaluation/reranking_development.json` for per-query results.

## Verification

Verified after complaint-context improvements and local reranking: **190 tests passed**, Ruff lint/format checks passed. Live API checks validated the reported broadband complaint, reranked evidence and empty filtered results. Earlier dataset-switch checks confirmed all three retrieval modes returned synthetic-only evidence. Nine smoke searches completed. A second indexing run skipped all 330 documents and generated zero new embeddings. Existing FastAPI/Starlette dependency deprecation warnings remain.

```powershell
python -m ruff check app scripts tests
python -m ruff format --check app scripts tests
python -m pytest tests/unit -q -p no:cacheprovider
$env:RUN_DB_TESTS="1"
python -m pytest -q -p no:cacheprovider
python -m scripts.check_retrieval --output data/evaluation/retrieval_report.json
```

Unit tests do not download models or require PostgreSQL. Integration tests use rollback-only schemas, plus a separate-session indexing-lock check; do not run them while indexing. Smoke queries exercise all three retrieval modes but are not a relevance benchmark. Test coverage includes corrupt artifacts, train/test leakage, synthetic evidence claims, invalid filters, duplicate rank candidates, SQL injection inputs, cache refresh, transaction rollback and indexing/retrieval exclusion.

## Project structure and progress

| Location | Responsibility |
|---|---|
| `app/api` | HTTP contracts and sanitized errors |
| `app/ingestion` | Evidence validation, synthetic generation and artifact integrity |
| `app/understanding` | Local classifier training, inference and explainable complaint signals |
| `app/retrieval` | Chunking, embeddings, lexical/semantic search and fusion |
| `app/database` | Connections, persistence and indexing checkpoints |
| `scripts` | Dataset preparation, database setup, indexing and demonstrations |
| `tests` | Unit and opt-in PostgreSQL integration tests |

| Phase | Status |
|---|---|
| Day 1: foundation and evidence semantics | Complete |
| Day 2 A–D: design, chunks, embeddings and indexing | Complete |
| Day 2 E–I: vector/BM25/fusion, API/CLI and verification | Complete |
| Dataset transition: telecom corpus and isolated index | Complete |
| Day 3a: local understanding | Implemented: `POST /api/v1/analyze`, training and analysis CLIs |
| Day 3b: local reranking | Implemented: optional `/api/v1/retrieve` reranking, CLI and development comparison |
| Day 3c: grounded drafting and citation validation | Planned |
| Day 4: reviewed evaluation and model comparisons | Planned |
| Later: evolving categories, operational metrics and UI | Planned |
