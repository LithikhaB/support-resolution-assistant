# Interview notes

## Problem and demo

This implements SynaptAI Use Case 2. Agents paste a complaint, see category/product/severity/sentiment, and get grounded troubleshooting steps with KB and resolved-case citations. The main demo should use the statement's evening-disconnection complaint, then a Wi-Fi-only fault, a billing issue, and a shared outage.

## Architecture

One Dockerized FastAPI microservice plus PostgreSQL/pgvector. Modules keep understanding, retrieval and drafting separate without adding distributed-service overhead. Root `architecture.svg` and the README show the current flow.

## Understanding

Groq/Gemini receives the full taxonomy and few-shot examples in one structured call. This corrects the old bug where a weak local classifier limited the LLM to the wrong shortlist. Explicit local impact rules remain explainable; quoted LLM severity/sentiment fill gaps. MiniLM + logistic regression is the local fallback. Its scores are not guaranteed confidence, and its thresholds are selected on dev data during setup.

All 30 prompt examples now come from training families; a provenance regression rejects held-out examples. Predicted category is advisory and cannot override a better-ranked applicable procedure. Extraction rules reject invented physical damage and unsupported calls-only observations. New class mappings are data-driven; local classifier support requires retraining rather than pretending the LLM taxonomy also updates model weights.

## Retrieval and grounded drafting

Dense embeddings handle paraphrases. PostgreSQL full-text search handles lexical matches through a shared GIN index. RRF combines ranks, avoiding incomparable score scales. In-memory BM25 remains an exploration baseline. Optional reranking is more expensive on CPU.

KB provides conditional troubleshooting procedures. Separately retrieved resolved cases explain what happened in similar cases, without treating that as the customer's confirmed diagnosis. Generated instructions need supplied source IDs and a grounding review. Invalid output or provider failure keeps the cited local plan. Diagnostic conditions are not withheld merely because they are unconfirmed; explicit contradictions still exclude inappropriate procedures.

## Simple useful behavior

Clarification comes with preliminary guidance. Completed actions are remembered, answered Ethernet checks are not repeated, and a shared outage receives incident guidance. The UI is a complaint form, plan, sources and follow-up field. Case storage, approval workflows and handoff management were removed from the submission scope.

## Data and evaluation

The statement permits synthetic data. This corpus is AI-authored and fictional-provider material; simulated outcomes are labelled honestly. Family-separated splits prevent paraphrases of one scenario appearing on both sides of a holdout. This does not make synthetic labels an independent customer benchmark.

Check offline regression tests, rollback-only DB integration tests, recorded natural complaints and provider/fallback telemetry. Keep local-only and live-provider results separate. Frozen historical retrieval/classifier ablations are exploration evidence, not fresh measurements of changed code. Manual ratings are not fabricated and reviewer independence is not automatically established.

## Evolution

Additive staging validates new records, preserves existing IDs and rejects split leakage. Indexing updates changed fingerprints. New KB does not need retraining. New classes enter the configured taxonomy for the LLM; train/dev examples and retraining support the local fallback. The DNS example runs in isolated artifacts and a separate database.

## Production choices

The image runs as non-root, the database and model cache are persistent, initialization is executable, and CI checks lint/tests and builds a downloadable API image. Timeouts, bounded concurrency, provider failover, index locks, readiness and sanitized errors protect the service. One worker limits CPU model memory; real replicas need a shared admission policy and appropriate resource limits. The supplied statement does not mandate a hosted deployment.

The API stays local by default. Before exposing it publicly, add gateway authentication, per-client rate limits and centralized admission. PostgreSQL's shared lexical index avoids a separate per-worker BM25 corpus. Replicas duplicate embedding/reranker memory; scale only after measuring CPU latency and memory. Persist metrics with an external Prometheus collector because counters reset on process restart. Keep indexing separate from request-serving replicas and use the existing index lock for updates.

For provider limits, pace evaluation and report quota failures explicitly. Cache repeated understanding/selection calls, keep each case to one relevant procedure, and retain a conditional local plan on failure. A model's grounding review is a quality check, not an independent guarantee. Never claim an adjustment, repair or external handoff has been performed.

## Honest limits

English-only extraction, fictional provider procedures, synthetic evaluation labels, CPU reranking cost and external provider quotas. Authentication and a curated provider KB are deployment-specific work. A Dockerfile is not proof of a successful build: verify the CI/container result on a Docker-capable host.
