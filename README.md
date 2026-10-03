# AI-Powered Telecom Support Resolution Assistant

An evidence-grounded ticket resolution assistant for support agents, implemented as a FastAPI service with a local agent workspace.

## Problem statement: what I understood

Agents receive incomplete complaints, changing answers and repeated troubleshooting attempts. Keyword search can miss equivalent symptoms, while a similar historical ticket can have a different cause. An assistant should help the agent find a supported next step without pretending it has diagnosed or repaired the service.

The selected problem statement requires:

- Understand intent/category, affected product, severity and sentiment.
- Retrieve relevant **resolved tickets and knowledge-base articles** using semantic search.
- Use an LLM to draft a step-by-step resolution with citations.
- Support evolving knowledge and ticket categories through a simple microservice design.

## What conventional solutions provide

- Keyword search or static scripts require agents to translate the complaint into search terms.
- A chatbot can produce fluent instructions without showing their source or diagnostic conditions.
- A retrieved similar case helps investigation, but its historical outcome does not confirm the current cause.

These describe the design problem, not measured claims about a particular commercial product.

## What this solution does

1. **Understand:** a locally trained classifier proposes a category; language extraction identifies quoted customer observations. Severity and sentiment remain separately explained assessments.
2. **Retrieve:** BM25 and MiniLM/pgvector search feed reciprocal rank fusion; an optional local cross-encoder reranks a bounded pool.
3. **Ground:** complete applicable KB procedures provide exact diagnostic gates, actions and restrictions. Resolved histories linked to those KB articles provide separate `T` citations.
4. **Draft:** Groq selects relevant supplied procedures and writes a short introduction. A model critique checks the introduction and choices; the application appends exact conditional steps and citations.
5. **Degrade gracefully:** **Groq → Gemini → extractive cited plan**. Invalid JSON, unsupported citations, faithfulness rejection, timeout or rate limit can trigger fallback. A short circuit avoids repeatedly calling a failing provider.
6. **Review:** agents clarify, edit, accept or reject drafts and record outcome evidence. Reviews retain the original response and generation; no repair, payment or external handoff is performed.
7. **Evolve:** validated new articles/tickets are staged additively. New classifier labels require train/development examples, retraining, evaluation and a category/service mapping, rather than a storage redesign.

Generative wording is marked for agent review. `validation` covers the original exact-source contract; `faithfulness_status=model_checked` describes a fallible model critique, not proof of correctness. The workspace offers “Review language draft” to save reviewed wording as an explicit agent edit.

## Architecture

![Project architecture](docs/architecture%20v1.svg)

The SVG shows the full project direction. The implemented request path is:

`API / CLI → RetrievalService → BM25 + pgvector → RRF → ranked evidence`

## Dataset and why it changed

The [Hugging Face support dataset](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) helped build the first ingestion/retrieval pipeline. Its broad domains and unverified replies did not provide the focused telecom diagnostic evidence needed for this challenge. The [GitHub Ticket_data alternative](https://github.com/santhoshmishra/Ticket_data) was a candidate, rather than evidence of verified telecom repairs. The engineering reason for initially using Hugging Face was its dataset interface; this project makes no unsupported claim that its resolution quality exceeds the GitHub data.

The active dataset is an authored **fictional telecom corpus**, not real provider policy:

| Artifact | Purpose |
|---|---|
| 60 diagnostic families / 15 categories | Distinct telecom investigation scenarios |
| 480 simulated resolved variants + 30 unresolved examples | Historical evidence with explicit outcome provenance |
| 60 conditional KB procedures | Diagnostic gate, action and restriction |
| 240 train / 120 development / 120 test queries | Scenario families remain separated |
| 330 indexed records | Training histories, unresolved histories and KB; development/test tickets are excluded |
| 16 natural complaint cases + declarative golden checks | Development conversation regression checks |

Synthetic outcomes stay `simulated_resolved`. Author-written relevance labels and golden expectations require independent support-agent review before they can be described as an independent quality benchmark.

## Phases and implemented endpoints

| Phase | Work | Progress |
|---|---|---|
| Foundation and dataset | Evidence schema; synthetic provenance; reproducible preparation | Complete |
| Retrieval | Chunking/indexing; BM25 + vectors + fusion; reranker and ablation | `POST /api/v1/retrieve` |
| Understanding | Local category model; quoted observations; impact/tone and prior attempts | `POST /api/v1/analyze` |
| Grounded resolution | KB and resolved histories; provider interface; cited conditional drafts | `POST /api/v1/resolve` |
| Conversation | Latest corrections; achievable clarification; separate issues | `POST /api/v1/conversation` |
| Human review | Saved cases; review/outcome history; local handoff download | `/api/v1/cases` and case follow-up/review/handoff routes |
| Evaluation and operations | Golden checks; privacy masking; fallback/cost counters; load smoke | `/api/v1/health`, `/ready`, `/metrics` |
| Knowledge evolution | Additive staging; reindexing; configurable category/service mappings | `scripts.stage_corpus_update` |

## Models and trade-offs

| Component | Implementation | Reason and limitation |
|---|---|---|
| Embeddings | `all-MiniLM-L6-v2`, 384 dimensions, CPU | Small local model; limited context length |
| Category model | Logistic regression over local MiniLM features | Trainable local component; scores are uncalibrated and can abstain |
| Reranker | `cross-encoder/ms-marco-MiniLM-L6-v2`, CPU | Optional relevance scoring; measured ablation does not show a universal improvement |
| Primary LLM | Groq `openai/gpt-oss-120b` | Remote understanding/drafting; account availability and quotas apply |
| Secondary LLM | Gemini `gemini-2.5-flash` | Independent provider fallback; also subject to quotas |
| Last fallback | Local extractive cited plan | Preserves sources and conditions; less flexible language understanding |

The previous Groq model was unavailable for the configured account; discovery and live checks confirmed the replacement. See [Groq model documentation](https://console.groq.com/docs/model/openai/gpt-oss-120b) and [Gemini structured output documentation](https://ai.google.dev/gemini-api/docs/structured-output).

The existing development reranker ablation reported expected-KB hit@5 of **50.0% for hybrid vs 36.7% with reranking** on the same candidate pools. This is a measured regression, so reranking remains configurable. Retrieval relevance, exact citation integrity and answer quality are measured separately.

## Run and test

Use the project virtual environment, PostgreSQL and the prepared local model/index artifacts. For a fresh setup, use Python 3.11 or 3.12, create `.venv`, install `requirements.txt`, copy `.env.example` to `.env`, and prepare the corpus/models/index with the scripts below. The working environment was also tested on Python 3.14; dependencies emit deprecation warnings there.

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Configure the existing `.env`:

```dotenv
LLM_ENABLED=true
GROQ_API_KEY=your_groq_key
GROQ_MODEL=openai/gpt-oss-120b
GEMINI_API_KEY=your_gemini_key
GEMINI_MODEL=gemini-2.5-flash
LLM_TIMEOUT_SECONDS=25
LLM_CIRCUIT_SECONDS=60
LLM_CACHE_SECONDS=120
MAX_RESOLUTION_REQUESTS=2
```

Keys stay in `.env`, which Git ignores. A blank/missing key skips that provider. `LLM_ENABLED=false` demonstrates the completely local fallback. Restart the server after configuration or model changes.

For the existing prepared project:

```powershell
python -m scripts.check_db
python -m scripts.check_index
.\run.ps1
```

Open `http://127.0.0.1:8000` for the home page, `/workspace` for tickets or `/docs` for Swagger. The launcher uses the project virtual environment to avoid the global-Python missing-dependency problem.

For dataset/model preparation when artifacts are missing:

```powershell
python -m scripts.prepare_synthetic
python -m scripts.chunk_documents
python -m scripts.setup_database
python -m scripts.index_documents
python -m scripts.train_understanding
python -m scripts.calibrate_routing
python -m scripts.prepare_reranker
```

Check each script's `--help` and the configured database before first-time preparation; indexing preserves existing IDs and rejects incompatible corpus/embedding profiles.

CLI demonstration and automated checks:

```powershell
python -m scripts.resolve_complaint "My broadband drops. Ethernet works. All wireless devices disconnect. I already restarted the router." --json
python -m ruff check app scripts tests
python -m ruff format --check app scripts tests
python -m pytest tests/unit -q -p no:cacheprovider
$env:RUN_DB_TESTS="1"
python -m pytest -q -p no:cacheprovider
python -m scripts.evaluate_response_quality --delay-seconds 15 --output data/evaluation/quality_runs/my_run.json
python -m scripts.review_quality_report data/evaluation/quality_runs/my_run.json --output data/evaluation/my_summary.json
python -m scripts.load_test --requests 6 --concurrency 2 --output data/evaluation/my_load.json
python -m scripts.demo_provider_fallback --fail groq
python -m scripts.demo_provider_fallback --fail both
```

Evaluation/load output paths must be new. Golden checks cover facts, service scope, question progression and unsafe wording; the report keeps manual quality ratings empty for human review. `review_quality_report --baseline previous_run.json` compares uncertainty, clarification and fallback rates only when the case packs and selected cases match. These are drift indicators, not a statistical production drift detector.

For knowledge updates and feedback:

```powershell
python -m scripts.stage_corpus_update new_records.jsonl --output data/synthetic/telecom_v2/processed
python -m scripts.chunk_documents --directory data/synthetic/telecom_v2/processed
$env:CORPUS_DIR="data/synthetic/telecom_v2"
python -m scripts.index_documents
python -m scripts.export_feedback --output data/workflow/feedback_review.jsonl
```

Review the staged corpus before activation. New categories need a valid train/development dataset with fresh checksums and separated families, retrained classifier/routing evaluation, plus an entry such as `{"new_wifi_category": ["home_wifi"]}` in `data/category_products.json`. Restart after publishing. Feedback exports are **curation candidates**, never automatically trusted resolutions or training labels.

Verified automated checks: **372 tests passed**, including rollback-only database integration tests. Ruff lint and format checks passed. Dependency deprecation warnings remain.

## Production considerations and measured limits

- **Privacy:** common emails, phone numbers and labelled account/customer/ticket identifiers are masked before remote calls. Quoted observations are restored locally for span validation. Masking is heuristic; names, addresses and unusual identifiers still need review. Local case files retain transcripts; this local demo has no multi-user access-control layer.
- **Guardrails:** untrusted data is separated from system instructions; schemas and exact quotes are checked. Generated citations must exist. Conditional repairs retain their source gates/restrictions. Model faithfulness critiques can still miss semantic errors, so human review remains required.
- **Latency and resilience:** timeout per provider call; Groq/Gemini failover; short circuit cooldown; bounded 128-entry short-lived cache; at most two expensive API requests per process by default. Overload returns 503 with `Retry-After`; health remains available.
- **Monitoring and cost:** `/metrics` exposes request latency/errors, provider tokens/timing, rate limits, fallback, cache and faithfulness rejection counts. Set optional `GROQ_INPUT_COST_PER_MILLION`, `GROQ_OUTPUT_COST_PER_MILLION`, `GEMINI_INPUT_COST_PER_MILLION`, `GEMINI_OUTPUT_COST_PER_MILLION` to current contracted rates; missing cost is unknown, not zero. Process counters reset at restart.
- **Scale:** PostgreSQL/pgvector is the shared retrieval index; model caches and language circuits are per process. Workers multiply model memory and provider traffic. SQLite cases are suitable for this local demo; multi-user deployment needs a shared transactional case store, authentication, connection pooling, distributed limits and durable metrics before claiming production readiness.
- **Evidence:** the local two-client load smoke completed 4/4 warm requests with passing source contracts. Cold latency was about 39.2 s; warm median 4.45 s and p95 14.48 s. Repeated inputs benefited from caching; these are not varied-traffic capacity figures.
- **Quality:** the five-case live provider-chain run passed 5/5 authored behavioural checks, with Groq, Gemini and extractive responses all observed. A broader 16-case run passed 15/16 and exposed a repeated billing-status question; that defect was fixed and its live regression retest passed. The final disabled-LLM run passed **16/16** authored behaviours. Both provider quotas were reached during the broader run, so 17 of its 26 issue snapshots used extractive fallback. These outcomes do not establish independent human-rated answer quality. Reports are preserved under `data/evaluation`.

The remaining submission work is independent support-agent review of golden answers and generated wording, plus a deployment-specific security/capacity validation if this prototype is exposed beyond the local interviewer demonstration. No provider contact directory or extra ticketing integration is included.
