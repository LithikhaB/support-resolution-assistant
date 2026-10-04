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

The grounded-resolution requirement remains partially satisfied in quality: the screenshot review exposed irrelevant cancellation and add-on procedures in a duplicate-payment response. Both generated and fallback plans need consistent relevance selection and concise presentation across categories. The supplied use case targets support agents, so the presentation should clearly distinguish customer checks from actions an authorized agent must perform.

The five authored HTTP checks cover selected categories, citations, questions and escalation. They do not establish that every displayed step is relevant, that all 15 categories are reliable, or that the wording is consistently helpful. Broader response-quality review remains necessary. LLM rate limits also caused local fallback in four of five responses in the final recorded run.

GitHub publication is pending the owner's commit/push. The CI workflow is configured but its remote execution has not been verified. Production-scale considerations are documented and several controls are implemented; production throughput and real-customer resolution accuracy have not been established. A public hosted deployment and independent human ratings are not explicit requirements in the supplied statement.
