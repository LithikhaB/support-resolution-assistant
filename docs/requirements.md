# SynaptAI challenge scope

Source: **Use Case for SSN.docx**, dated 29 September 2026, Use Case 2; and **Evaluation.docx**, supplied by the repository owner. These are the submission requirements, rather than the earlier informal feedback scores.

| Requirement | Implementation | Verification |
|---|---|---|
| Parse intent/category, product, severity and sentiment | One structured Groq/Gemini understanding call with full taxonomy and exact complaint quotes; local MiniLM classifier and rules remain available when providers fail | Understanding regressions and provider-backed complaint runs |
| Retrieve similar resolved tickets and KB articles semantically | Local embeddings + pgvector, shared PostgreSQL lexical ranking, RRF and optional reranking; resolved history retains its outcome provenance | PostgreSQL integration tests, frozen retrieval comparisons |
| LLM draft with grounded, step-by-step guidance and citations | LLM writes steps from retrieved conditional procedures; source IDs are checked, a grounding review rejects unsupported instructions, and failures retain a cited local plan | Provider fallback and citation tests; recorded natural complaints |
| Evolving data and ticket classes | Validate and stage additive JSONL updates; incremental indexing; configurable category mapping; isolated new-category demonstration | Corpus-update tests and evolution demonstration |
| Architecture diagram | Root architecture.svg and README Mermaid | Diagram matches the current request path |
| Full executable code in GitHub | API, UI, setup CLIs, Dockerfile, Compose and CI workflow | Local checks; CI runs when code is pushed |
| Additional exploration | Existing retrieval ablations and classifier comparisons, clearly labelled synthetic development measurements | Preserved evaluation reports |
| Evals on system health | /health, /ready, request/provider metrics, offline regressions, opt-in DB integration and varied-input load tool | API checks and reproducible commands |
| Production-scale considerations | Non-root container, external database, shared lexical index, timeouts, bounded concurrency, provider fallback, sanitized errors and fingerprinted ingestion | Tests plus Docker build job |

The statement explicitly permits **open-source or synthetic data representing real-world data**. It does not explicitly require a hosted deployment, a customer account system, case management, or independently verified human ratings. Synthetic test results must still be labelled honestly. No part of this application executes telecom repairs or confirms a live network diagnosis.

The evaluation weights are problem understanding 15%, solution depth/production scale 25%, design decisions 20%, code 25%, and checkpoints/evals/monitoring 15%.

## Completion audit

Rechecked against both supplied DOCX documents on 4 October 2026. The core parsing, retrieval, data evolution, architecture, Docker runtime and health-evaluation components are implemented and have local verification evidence. This does not establish complete answer quality or production capacity.

The grounded-resolution flow is implemented. Category forcing has been removed from procedure selection, and stage tracing records where evidence is lost. The owner's five-answer review led to corrections for repeated checks, support-task labels, unsupported menus, diagnostic conditions and historical citations. Those corrections pass regression tests. Before the final fallback-ranking correction, offline runs retained the expected family KB in only 58.3% of development and 52.5% of test plans; this remaining selection loss limits any claim of complete answer quality. The supplied use case targets support agents, so customer checks and authorized support actions are distinguished.

The current automated suite passes 481 tests, including database integration. Owner ratings are preserved in answer-review.md and recorded separately from model grounding reviews. The fresh live development run produced nine generated answers in 30 attempts; quota and validation failures prevented the 20-answer target. The final owner-review wording corrections still need renewed provider-backed review. Dataset expansion is deferred at the owner's request.

Remote CI succeeded for published commit cd8f8a2; the current working changes await the owner's commit/push and a new CI run. Production-scale considerations and tradeoffs are documented, with several controls implemented. Production throughput and real-customer resolution accuracy have not been established. A public hosted deployment and independent human ratings are not explicit requirements in the supplied statement. See verification.md for evidence and limits.

Final Docker owner-review smoke: five successful responses, all provider-limited fallbacks; four expected procedures, with periodic-disconnection relevance still open. Preserving reranker order raised development expected-KB retention to 85/120 (70.8%). See the recorded source fingerprints before comparing reports across versions.
