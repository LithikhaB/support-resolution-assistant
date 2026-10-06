# Telecom Support Ticket Resolution Assistant

> **Use Case 2.** An agent pastes a customer complaint and gets back the **category, product, severity, sentiment, steps already tried, relevant KB procedures, similar resolved tickets**, and a **cited, conditional troubleshooting plan**.

**Scope:** the assistant *proposes* checks for **agent review**. It does **not** diagnose live networks, run repairs, issue refunds or create handoffs.
**Data:** all telecom evidence is **AI-authored synthetic** and **not expert-reviewed**.

---

## Architecture

![Request, evidence and evolution](architecture.svg)

| # | Phase | What happens |
|---|-------|--------------|
| 1 | **Validate & admit** | Pydantic contracts, **credential scrubbing**, per-IP and global **admission budgets** |
| 2 | **Understand** | MiniLM **classifier** plus quoted-span rules give category, product, severity, sentiment, actions tried and facts |
| 3 | **Retrieve** | **Hybrid search** (pgvector + PostgreSQL FTS, fused with **RRF**) plus optional **reranker** over KB articles and resolved tickets |
| 4 | **Draft** | Local **conditional plan** with exact-source **citations**; skips completed checks and keeps restrictions |
| 5 | **Reword (optional)** | **Groq** drafts, **Gemini** critiques; any failure falls back to the local plan |

---

## Quick Start

### Docker (recommended)

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
# Optional: set GROQ_API_KEY / GEMINI_API_KEY, or LLM_ENABLED=false for local-only plans
docker compose --profile app up --build -d
docker compose exec -T api python -m scripts check
```

Then open the **console** at http://127.0.0.1:8000/ or **Swagger** at http://127.0.0.1:8000/docs.

### Local Python

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
docker compose up -d db
python -m scripts setup
python -m scripts chunk
python -m scripts index
python -m scripts train
python -m scripts calibrate
python -m scripts reranker
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

> `data/models` is git-ignored, so **train and calibrate are required** after a fresh clone.

---

## Commands

Run as `python -m scripts <command>`.

| Command | Purpose |
|---------|---------|
| `prepare` | Build the **synthetic corpus** |
| `chunk` | Split documents into overlapping chunks |
| `index` | Embed chunks and upsert into **pgvector** |
| `check` | Verify **index readiness** and embedding norms |
| `train` | Fit the **category classifier** on the dev split |
| `calibrate` | Choose **score/margin thresholds** on dev |
| `reranker` | Warm the **cross-encoder** |
| `analyze` | Run understanding on one complaint |
| `resolve` | Run the **full pipeline** on one complaint |
| `evaluate` | **Frozen** dev/test evaluation |
| `load` | **Load test** against a running API |
| `demo` | Quick offline demo with verification |
| `evolve` | Isolated **new-class demo** in a fresh DB |

---

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/v1/health` | Process **liveness** |
| `GET` | `/api/v1/ready` | DB schema and **index state** |
| `POST` | `/api/v1/analyze` | Category, product, severity, sentiment, facts |
| `POST` | `/api/v1/retrieve` | Ranked KB and resolved-ticket **evidence** |
| `POST` | `/api/v1/resolve` | **Cited plan** for one complaint |
| `POST` | `/api/v1/conversation` | Same flow with bounded customer follow-ups |
| `GET` | `/api/v1/metrics` | JSON latency, error and provider telemetry |
| `GET` | `/metrics` | **Prometheus** counters and histograms |

---

## Results

Retained **v3.1** dev/test reports. All runs use the local fallback only, PostgreSQL lexical search and the published MiniLM classifier.

| Metric | Dev | Test |
|--------|-----|------|
| **Category top-1 accuracy** | 76.67% | **83.33%** |
| **Macro-F1** | 0.7551 | 0.8077 |
| **KB hit@5** (hybrid) | 90.00% | 95.83% |
| **KB hit@5** (reranked) | 93.33% | **100.00%** |
| **Citation contract passed** | 100.00% | **100.00%** |
| Severity correct / unknown | 40.00% / 43.33% | 13.33% / 83.33% |
| Sentiment correct / unknown | 75.00% / 25.00% | 100.00% / 0.00% |
| Latency p50 / p95 (ms) | 4666 / 6282 | 6996 / 16658 |

*Citation validity is not diagnostic correctness, and synthetic sentiment does not prove language robustness.*

### Load test (historical, offline extractive)

| Concurrency | p50 / p95 / p99 (ms) | Accepted | Success RPS | Error rate |
|-------------|----------------------|----------|-------------|------------|
| 1 | 145 / 182 / 189 | 12/20 | 4.61 | 40% |
| 5 | 133 / 292 / 381 | 3/20 | 4.43 | 85% |
| 20 | 480 / 527 / 532 | 2/20 | 3.04 | 90% |

*The first cold request takes about 20 s. Fast rejections inflate RPS, so this is **not** production capacity. No provider calls were recorded.*

---

## Evolving Data Demo

```powershell
python -m scripts evolve --output .work/dns-evolution-new.json
```

Runs in a **separate corpus and database**, then checks that:
- the **new class** is registered and predicted
- the **new article** is cited
- the **citation contract** still passes and the index is ready
- **old classes** are preserved and old dev accuracy stays within **5 points**

---

## Limitations

- **Synthetic data only:** a fictional provider, not expert-reviewed
- **Weak severity:** rules and classifier often return *unknown*
- **LLM needs keys:** without Groq/Gemini keys the assistant returns the **extractive local plan**
- **No production auth:** the anonymous owner cookie and admin key are not real authentication
- **Citations are not diagnoses:** exact-source checks pass, but semantic correctness is not measured

---

## Repository Map

```
app/api           Contracts and endpoints
app/understanding Classifier, calibration, signals, LLM extraction
app/retrieval     Dense and lexical search, RRF, diversity, reranking
app/resolution    Cache, history, applicability, selection, plan, validation, policy
app/llm           HTTP, privacy, provider circuits, budgets, caches
app/ingestion, db Validation, indexing, publication, schema
app/monitoring    Admission, metrics, budgets
app/web           Agent console (index.html, app.js, style.css)
scripts/          CLI commands
data/synthetic/telecom_v3_1  Active corpus (62 KB, 240 resolved, 30 historical)
data/evaluation   Frozen reports, gate thresholds, load report, challenge cases
docs/design-decisions.md     Key architectural decisions
```