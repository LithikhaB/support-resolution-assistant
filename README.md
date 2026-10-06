# Telecom Support Ticket Resolution Assistant

A FastAPI service for a telecom support agent: paste a complaint, identify its category, product, severity and sentiment, then review a conditional cited plan. Repairs require agent review.

![Architecture](architecture.svg)

## Pipeline

1. Parse the complaint and follow-ups with a local classifier and quoted context rules; optional LLM extraction supplies validated observations.
2. Check revision-keyed evidence caches and reviewed recent outcomes; retrieve applicable KB and past cases through MiniLM, PostgreSQL full-text search, RRF and optional reranking.
3. Build ordered local guidance, acknowledge completed actions, and keep repairs conditional on diagnostic findings.
4. Optionally use Groq/Gemini for wording and critique; exact citations and protected steps must pass validation or the local plan is returned.
5. Return decision, priority, target, field evidence, cited steps and honest provider/validation badges to the agent console.

## Quick start: Docker

Requires Docker Desktop/Engine with Compose. Copy `.env.example` to `.env` if absent, keep credentials private, and set `LLM_ENABLED=false` for a local demo.

```sh
docker compose --profile app up --build -d
docker compose exec api python -m scripts check
```

Open [the console](http://127.0.0.1:8000) or [API docs](http://127.0.0.1:8000/docs). Startup initializes PostgreSQL/pgvector, indexes the supplied corpus, trains missing classifier/routing artifacts and warms models. First startup downloads models; PostgreSQL, data and model caches use persistent volumes. Never delete volumes to upgrade.

## Quick start: local Python

Requires Python 3.12+, PostgreSQL/pgvector and optional Node.js for UI tests. Create `.env` as above.
Activate `.venv/Scripts/Activate.ps1` on Windows or `source .venv/bin/activate` on Unix.

```sh
python -m venv .venv
# Activate .venv before the remaining commands.
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
docker compose up -d db
python -m scripts setup
python -m scripts prepare
python -m scripts chunk
python -m scripts index
python -m scripts train
python -m scripts calibrate
python -m scripts reranker
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Keep one indexing runtime per database. When changing library versions, use `python -m scripts index --rebuild`; configuration fingerprints reject incompatible embeddings. Setup/training use train/dev, not test.

## Commands and verification

```sh
python -m ruff check app scripts tests
python -m ruff format --check app scripts tests
python -m pytest tests/unit -q
python -m scripts.eval_gate
node --test tests/web/agent_console.test.cjs
```

For PostgreSQL integration tests, set `RUN_DB_TESTS=1` and run `python -m pytest -q`.
For the Docker gate: `docker compose exec api python -m scripts.eval_gate`.
The gate requires prepared/indexed data and trained/calibrated artifacts, runs only
frozen dev queries with LLMs disabled, and fails on classifier, KB-retention or citation regressions.

`python -m scripts --help` lists setup, prepare, chunk, index, check, train, calibrate,
reranker, analyze, resolve, converse, evaluate, audit, quality, load, demo, ingest-demo,
stage, evolve and legacy publication commands. Evaluation/demo output names must be new:

```sh
python -m scripts evaluate --split dev --output .work/dev.json
python -m scripts quality --output .work/quality.json
python -m scripts demo --output .work/demo.json
python -m scripts load --requests 20 --concurrency 5 --output .work/load.json
```

## Endpoints

All application endpoints below use `/api/v1`; metrics are at `/metrics`.

| Method | Path | Purpose |
|---|---|---|
| POST | `/analyze`, `/retrieve` | Parsed fields or ranked evidence |
| POST | `/resolve`, `/conversation` | Cited plan; up to 4 issues and 8 issue-linked follow-ups |
| GET | `/health`, `/ready` | Process liveness and model/index readiness |
| POST | `/ingest` | Up to 20 validated documents; requires `X-Admin-Key` |
| GET | `/categories` | Deployed classifier classes and document counts |
| GET | `/conversations`, `/conversations/{id}` | List/read stored conversations |
| DELETE | `/conversations/{id}` | Delete a stored conversation |
| POST | `/conversations/{id}/review` | Record an agent-reviewed outcome |
| GET | `/metrics` (no prefix) | Prometheus counters and stage timing |

Example resolve body: `{"query":"My broadband drops every evening; I restarted the router twice."}`. Send `turns` with `/conversation`.

## Retained evaluation evidence

These are the frozen `v31_postgres_{dev,test}_release_20261005.json` snapshots;
they predate the later category-based severity fallback and are not fresh quality claims.
Each split contains 120 queries across 15 held-out scenario families; LLMs were disabled.

| Metric | Dev | Test |
|---|---:|---:|
| Published category accuracy | 76.67% | 83.33% |
| Category macro-F1 | 0.7551 | 0.8077 |
| Expected-KB hit@5, KB-focused | 90.00% | 95.83% |
| Expected-KB hit@5, reranked | 93.33% | 100.00% |
| Citation contract validity | 100.00% | 100.00% |
| Severity correctness | 40.00% | 13.33% |
| Comparison latency p50 / p95 | 4,666 / 6,282 ms | 6,996 / 16,658 ms |

Latency covers the multi-strategy evaluation comparison, not an API SLA. Citation
validity checks exact source support; it does not establish diagnostic correctness.
The retained `v31_challenge_confirmed_20261005.json` contains 53 completed authored
cases and 66 snapshots; human ratings are absent, so semantic plan quality is unscored.

`load_test_report.json` records 20 warm repeated-query requests per concurrency:

| Concurrency | p50 / p95 / p99 ms | All / successful RPS | Rejection/error rate |
|---|---:|---:|---:|
| 1 | 145 / 182 / 189 | 7.68 / 4.61 | 40% |
| 5 | 133 / 292 / 381 | 29.53 / 4.43 | 85% |
| 20 | 480 / 527 / 532 | 30.40 / 3.04 | 90% |

This historical run used earlier admission limits and shared budgets across runs; RPS/latency include 429/503 rejections. It is not a production-capacity claim.

## Evolving data

```sh
python -m scripts evolve --output .work/evolution.json
```

This uses `data/evolution/dns_category_demo.json`, a separate corpus/database, training,
calibration, incremental indexing and regression checks; active artifacts stay unchanged.
For live KB updates, set a private `INGEST_ADMIN_KEY`, recreate the API, then run
`python -m scripts ingest-demo`. Empty keys disable ingestion (403). Valid updates are
published under the index lock; failures roll back files/index and revisions invalidate
caches. Retrieval updates immediately. New category prediction also needs labeled
train/dev examples, a product mapping, `python -m scripts train` and `python -m scripts calibrate`.

## Limitations and repository map

Data is AI-authored synthetic: 62 KB articles, 240 simulated resolved training tickets and 30 historical responses. No simulated outcome is a real repair. Severity remains estimated; sparse context, language variation and ambiguous intent need review.
LLM wording needs provider keys and quota; missing keys, throttling or invalid output
return a labelled extractive plan. Generated guidance cannot confirm a diagnosis.
CPU reranking, model memory and cache/storage coordination need measured capacity planning for replicas; admission limits and caches alone do not prove production scale.

`app/` contains API/config/database, understanding, retrieval, resolution, LLM,
ingestion, evaluation, monitoring and the web console. `scripts/` holds executable
setup/evaluation/demo tools; `tests/` holds unit, DB and UI checks; `db/` holds schema
and migrations. `data/synthetic/telecom_v3_1` is active; older corpora/generators remain
because preparation and regression tests reference them. `data/evaluation` keeps
required inputs and final evidence. See [design decisions](docs/design-decisions.md).
