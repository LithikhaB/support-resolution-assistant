# Submission verification — 4 October 2026

The local Docker image was built and started against the existing PostgreSQL volume. Both services became healthy. The API readiness endpoint returned HTTP 200. Index verification found 330 documents, 330 chunks and 330 normalized embeddings, with a valid cosine HNSW index and no invalid embeddings.

The first container startup regenerated the index because the Linux runtime had a different configuration fingerprint. A subsequent rebuilt-container startup skipped all 330 unchanged documents and generated zero embeddings. Source rows and stable chunk IDs are preserved; the integration regression also checks interrupted rebuild recovery and rejection of an incomplete input corpus.

The final automated suite passed **470 tests**, including PostgreSQL integration checks. Ruff and Git whitespace checks passed. These are individual assertions/cases across 35 test modules, not hundreds of separate test scripts.

Five synthetic complaints were submitted through the running Docker HTTP API: intermittent broadband, Wi-Fi coverage, duplicate settled payments, rejected number transfer and a shared outage. All five final authored behavior checks passed. Billing no longer repeats the supplied payment status; porting no longer asks a generic mobile-service question or includes an outgoing-call restriction procedure.

Subsequent screenshot review found a remaining relevance defect: the duplicate-payment plan also displayed unrelated cancellation and add-on procedures. The checks above did not test that exclusion. Passing them must not be described as complete response-quality verification. Consistent concise presentation and relevant step selection across categories remain unfinished.

- [Final HTTP behavior summary](../data/evaluation/docker_final_http_summary_20261004.json)
- [Final full response evidence](../data/evaluation/quality_runs/docker_final_http_20261004.json)
- [Earlier Docker provider run](../data/evaluation/docker_http_summary_20261004.json)
- [Isolated new-category evolution verification](../data/evaluation/evolution_verified_20261004.json)

The earlier Docker run generated four of five responses with Groq/Gemini. The final repeat generated one with Groq and used the cited local fallback for four. Provider quota, availability and grounding failures remain visible in each response; successful HTTP requests must not be reported as successful LLM generations. These small development checks are not independent human answer-quality ratings or production performance measurements.

CI runs lint, the full database-backed test suite, an image build, fresh container startup/readiness checks, and Docker-image artifact export. Its remote execution status must be checked separately from the successful local checks above.
