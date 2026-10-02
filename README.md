# AI-Powered Support Resolution Assistant

**Intelligent Support Ticket Resolution Assistant — Synapt assessment**

## 1. Problem statement and my understanding

A telecom support agent receives a raw complaint such as:

> “My broadband drops every evening around 8. I have restarted the router twice, and I work from home.”

My understanding is that the agent needs a useful next action supported by evidence, not just a list of tickets containing the word “router”. The assistant must address these problems:

- **Different words, similar issues:** “internet keeps dropping” and “intermittent broadband disconnection” may describe the same symptom.
- **Important context:** the time pattern, affected service, business impact and troubleshooting already attempted should influence the response.
- **Manual triage:** intent, product, severity and sentiment must be interpreted from an unstructured complaint.
- **Trust in recommendations:** each proposed step should have a traceable source, and a historical reply must not be mistaken for a proven fix.
- **Changing knowledge:** new ticket categories and updated procedures must become usable without redesigning the system.

The target is a small service that retrieves relevant past tickets and KB articles, then drafts a step-by-step, cited resolution. Missing information or weak evidence should lead to clarification or escalation.

## 2. What the current solution has

**Existing workflow described in the assessment**

- Agents search past tickets and knowledge articles using keywords.
- Agents manually interpret the complaint, compare possible matches and select the next steps.
- Exact terms are useful, but wording differences can hide relevant evidence.

**What this implementation has today**

- Validated ingestion, source provenance, reproducible data preparation and a PostgreSQL schema.
- Token-aware complaint chunks, with historical replies linked through the parent document.
- Phase C CPU embedding service and an executable model smoke test; verification results are listed below.
- FastAPI liveness/readiness endpoints. Search, reranking and resolution generation remain planned.

## 3. What our solution does

1. **Prepare trustworthy evidence — implemented:** clean records, preserve answer variants and keep unknown outcomes explicit.
2. **Represent complaint meaning — implemented through Phase C:** create bounded chunks and normalized 384-dimensional local embeddings.
3. **Retrieve complementary matches — planned:** combine BM25 exact-term matches with pgvector semantic search using Reciprocal Rank Fusion, rather than adding incompatible raw scores.
4. **Understand and refine — planned:** extract complaint context and rerank candidates before building the evidence set.
5. **Recommend a justified action — planned:** generate cited steps, account for attempted troubleshooting, and request clarification or escalate when needed.
6. **Adapt and measure — planned:** handle data updates and evolving categories, compare retrieval approaches and monitor latency, failures and evidence quality.

## Architecture
![Architecuture Diagram](docs/architecture%20v1.svg)

## 5. Dataset and selection rationale

I chose [Tobi-Bueck/customer-support-tickets on Hugging Face](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) because its customer-message and agent-response structure fits the complaint-to-evidence workflow. 

|  | Hugging Face dataset | GitHub dataset |
|---|---|---|
| Text structure | Subject, complaint body and agent answer | Complaint type, descriptor and resolution description; no separate customer-message body column |
| Domain fit | Mixed customer/IT support topics | Municipal complaints, including noise, parking, rodents and sanitation |
| Useful source labels | Type, queue, priority, language and tags | Agency, borough, complaint type and status |
| Outcome evidence | A reply is not proof of a completed resolution | Includes status and closed date; resolution descriptions are present for 22,760 of 25,921 rows |
| Fit for this prototype | Better suited to natural-language complaint matching and response evidence | Useful for civic-ticket routing and lifecycle analysis |

**Reasons for the choice**

- Paired complaint/response text supports retrieval now and evidence assembly later.
- Source labels support filtering and selected evaluation tasks without inventing labels.
- The Hugging Face loader supports a repeatable download pinned to a source revision.

**Current local corpus**

| Measure | Count |
|---|---:|
| English source rows | 28,261 |
| Cleaned historical responses | 23,790 |
| Exact complaint groups | 23,643 |
| Retrieval chunks | 23,811 |
| Documents needing multiple chunks | 21 |

**Limitations we preserve explicitly**

- This is mixed-domain, templated data, not a reviewed telecom corpus. Public availability does not establish that tickets are real customer incidents.
- All current records have `outcome_status=unknown`; `response` stores the agent reply and `resolution` remains null. Source ticket type/priority are distinct from inferred intent/severity.
- Product, sentiment, intent and severity are not populated without evidence. No KB collection is present yet.
- The dataset card lists **CC BY-NC 4.0**. GitHub's closure/status fields are an advantage it retains; our choice does not prove superior retrieval quality.
- The legacy local snapshot's upstream revision is unknown. File hashes identify it; future downloads record an immutable revision. Near-duplicate leakage still needs evaluation controls.

## 6. Implementation phases

Each row lists the phase's two main outcomes. Day 2 is deliberately limited to retrieval foundations.

| Stage | Scope | Status |
|---|---|---|
| Day 1 — foundation | Normalize records and track provenance; validate evidence and database readiness | Complete |
| Day 2 A — inspection | Inspect data/schema; agree retrieval design | Complete |
| Day 2 B — chunking | Preserve short complaints; split long text within token limits | Complete |
| Day 2 C — embeddings | Reuse a local CPU model; validate batched document and query vectors | Complete |
| Day 2 D — indexing | Transactionally load documents/chunks; build cosine HNSW index | Planned |
| Day 2 E — vector search | Search pgvector with parameterized SQL; return distances and evidence | Planned |
| Day 2 F — BM25 | Index the same chunk corpus; retrieve exact technical terms | Planned |
| Day 2 G — fusion | Merge candidates using RRF; expose ranks and source contributions | Planned |
| Day 2 H — API and CLI | Add bounded retrieval requests; demonstrate ranked results | Planned |
| Day 2 I — verification | Test retrieval/failure paths; run clearly labeled smoke queries | Planned |
| Day 3 — understanding and RAG | Extract context and rerank evidence; draft cited steps with clarification/escalation | Planned |
| Day 4 — evaluation | Build reviewed query sets; compare retrieval and answer-grounding quality | Planned |
| Day 5 — operations and UI | Add ingestion/update behavior and runtime metrics; expose an agent interface | Planned |
| Day 6 — final demonstration | Verify reproducible setup; present results and limitations | Planned |

## 7. Phase-wise endpoints and progress

| Phase | Working interface | What it provides |
|---|---|---|
| Day 1 | `GET /api/v1/health` | Process liveness |
| Day 1 | `GET /api/v1/ready` | PostgreSQL/schema readiness; 503 when unavailable/incomplete |
| Day 1 | `GET /docs` | FastAPI interactive documentation |
| Day 1 | `python -m scripts.prepare_data` | Documents JSONL and quality/checksum manifest |
| Day 2 B | `python -m scripts.chunk_documents --preview 2` | Chunk JSONL, manifest and optional preview |
| Day 2 C | `python -m scripts.check_embeddings` | Real-model CPU check for dimensions, normalization and reuse |
| Day 2 D | `python -m scripts.index_documents` **planned** | Batched database indexing |
| Day 2 H | `POST /api/v1/retrieve` **planned** | Ranked evidence, not an LLM answer |
| Day 3 | `POST /api/v1/tickets/resolve` **planned** | Structured analysis and cited resolution draft |

### Run and test Phase C

From the repository root with the virtual environment activated:

```powershell
python -m pip install -r requirements.txt
python -m pytest tests/unit/test_embeddings.py -q -p no:cacheprovider
python -m scripts.check_embeddings
python -m pytest -q -p no:cacheprovider
```

- The first smoke test downloads the pinned model into `data/models/`; inference runs on CPU. No complaint text is sent to an embedding API.
- Subsequent runs can be offline: `$env:EMBEDDING_LOCAL_FILES_ONLY="true"`. If the cache is incomplete, unset it or use `"false"` once with network access.
- `EMBEDDING_BATCH_SIZE=32` is a conservative CPU starting point. Documents and queries use the same model revision as the Phase B tokenizer, normalized vectors and a 256-token ceiling. Overlong input is rejected instead of silently truncated.
- Model/service reuse is per process. Unit tests inject fake models and never download weights. Corpus-wide vector generation and database writes belong to Phase D.

### Foundation setup and verification

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
# First setup only; preserve an existing .env
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
# Configure PostgreSQL values in .env, then start the database if needed
docker compose up -d
python -m scripts.migrate_db
python -m scripts.check_db
uvicorn app.main:app --reload
```
