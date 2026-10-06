# Design decisions

## One service and one database

FastAPI keeps understanding, retrieval, resolution and ingestion in separate modules.
PostgreSQL stores tickets, chunks, conversations and shared budgets; pgvector adds
semantic search without another datastore. One CPU worker bounds model memory.
Replicas need measured CPU/memory capacity and coordinated storage/admission.

## Hybrid evidence, not a guessed diagnosis

MiniLM captures paraphrases; PostgreSQL full-text search retains exact terms. RRF
combines ranks without treating incompatible scores as comparable. Optional
cross-encoder reranking improves ordering at a CPU cost. The `bm25` compatibility
label means PostgreSQL full-text ranking when that backend is configured; in-memory
BM25 remains an exploration baseline. Similar resolved cases suggest investigations,
not proof that the current customer has the same fault.

## Validated local plan and optional language models

Understanding keeps quoted evidence separate from estimated category/severity defaults.
Completed customer actions and structured follow-ups constrain later steps. Explicit
contradictions exclude procedures; missing diagnostic findings keep repairs conditional.
The example deployment uses Groq for wording and Gemini for critique. With split
review disabled, generation uses Groq-to-Gemini failover and same-provider critique.
Optional extraction and selection are disabled in the example configuration. Exact source quotes, citations, protected actions and
restrictions remain
locally validated. Provider limits, invalid text or missing keys return the labelled
extractive plan. No repair, refund or external handoff is performed by the assistant.

## Caches and recent conversations

Short process caches and bounded daily JSON caches use input/evidence/model revisions.
Recent complaints supply context; only agent-reviewed outcomes are reusable evidence.
A similar complaint never makes an old diagnosis true for a new customer. The published
index revision invalidates evidence/solution lookup after ingestion. PostgreSQL
admission budgets, bounded request queues, provider backoff and circuit breakers limit
resource use; they do not establish a throughput guarantee.

## Synthetic data and evaluation

The active v3.1 corpus is AI-authored and outcomes remain `simulated_resolved`.
Family-separated train/dev/test files avoid paraphrase leakage. Train/dev select
classifier and routing settings; test is a held-out evaluation, not a tuning input.
The frozen dev gate enforces category, expected-KB and citation thresholds. Exact
citation validity and authored contract checks do not replace human plan-quality
review. Release reports are historical snapshots; severity remains a weak signal.

## Evolving knowledge and classes

Live ingestion validates IDs, KB references and split exclusions, stages artifacts,
then publishes an incremental index under the writer lock. Ordinary indexing,
publication and transaction failures restore the old artifacts/index. New articles
are immediately searchable; a new class also needs labeled examples, a product
mapping and train/calibrate before local category prediction can recognize it.
The CLI DNS evolution demo uses a separate corpus and database and checks regression
of existing classes. Legacy generators remain because preparation and tests use them.

## Operational boundaries

Health, readiness and Prometheus metrics expose service/index/provider conditions.
Scraped metrics need external storage because process counters reset on restart.
The console is an agent advisory tool. Authentication beyond the ingest admin key,
curated provider procedures, independent language/quality evaluations and production
capacity testing depend on the deployment and remain explicit limitations.

The [README](../README.md) is the implementation/command/measurement reference.
[Ideas](ideas.md) records considered approaches and future work. Ordinary rollback
is not a guarantee of atomic filesystem/database recovery after abrupt host failure.
