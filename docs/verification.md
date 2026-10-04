# Submission verification — 4 October 2026

The current source passes **481 tests**, including PostgreSQL integration tests. Ruff passes. These are test cases across existing modules, rather than 480 separate scripts. Checks cover retrieval, conditional drafting, citations, provider failures, privacy, indexing recovery and owner-review corrections.

## Owner review and corrections

[The owner's five-answer review](answer-review.md) is preserved verbatim. [Its scored record](../data/evaluation/owner_answer_review_20261004.json) is an owner review of the earlier answers, not an independent review or a rating of the corrected version. Mean scores out of two: context 1.6, relevance 1.8, grounding 1.6 and clarity 1.4.

Corrections now pass regression checks: free-form “Already checked” text reaches drafting; support investigation is distinguished from customer checks; prompts prohibit invented device menus and require every diagnostic condition; unrelated historical comparisons are excluded from generated citations; an anger-specific billing warning is omitted for calm complaints. Physical damage requires explicit damage wording, and post-transfer call failures retain number-porting procedures. Regression checks establish these behaviors, but do not establish that every new provider answer obeys them.

## Reproducible evaluation

Six held-out few-shot examples were replaced with training examples. The audit checks prompt provenance and traces candidate retrieval, applicability, selection and final citations. Predicted category no longer forces selection over retrieved evidence.

The recorded ablations use the current PostgreSQL full-text backend and predate the final fallback-ranking correction. It is **not BM25**, despite a compatibility score-field name. These measurements explicitly disable LLM calls and exercise the application's current resolution service.

| Measurement | Development | Test |
|---|---:|---:|
| KB reranked retrieval hit@5 | 90.0% | 100.0% |
| Expected family KB in final plan | 58.3% | 52.5% |

Reports: [development ablation](../data/evaluation/corrected_postgres_dev_20261004.json), [test ablation](../data/evaluation/corrected_postgres_test_20261004.json), [development stage trace](../data/evaluation/corrected_stages_dev_20261004.json), [test stage trace](../data/evaluation/corrected_stages_test_20261004.json). One expected KB per family is a strict metric; alternative contrast procedures can be useful. [Preserving the reranked order](../data/evaluation/final_ranking_dev_20261004.json) subsequently retained the expected KB in 85/120 development plans (70.8%), up from 58.3%. A title-aware experiment was reverted after worsening a reviewed mobile case. Remaining retrieval-to-selection loss is measurable and must not be presented as solved completely.

[The recorded live development run](../data/evaluation/corrected_live_dev_v2_20261004.json) made 30 attempts and produced **nine LLM-generated answers** with passing citation contracts and model grounding reviews. Provider quotas, malformed replies and grounding rejections caused fallback. The requested 20-answer target was not met. This run predates the final owner-review wording corrections; those corrections still need renewed provider-backed review when quota is available. A model grounding review is not a human quality rating.

## Docker, evolution and CI

Earlier Docker checks confirmed a ready index with 330 documents, chunks and normalized embeddings. Rebuilt-container startup preserved source rows and chunk IDs and skipped unchanged documents. The integration suite verifies interrupted rebuild recovery and rejects incomplete corpora.

[The isolated category demo](../data/evaluation/evolution_verified_20261004.json) adds a DNS article and class, predicts and cites it, and leaves the active index untouched. New classes require retraining and calibration for local outage resilience. [The fresh regression demo](../data/evaluation/evolution_regression_20261004.json) passed all seven checks. Original-category accuracy remained 92/120 (76.7%) before and after extension; all 15 original classes and active artifacts were preserved.

Remote CI for published commit `cd8f8a2` succeeded: [GitHub Actions run](https://github.com/LithikhaB/support-resolution-assistant/actions/runs/37204485529). This does not verify uncommitted changes. The owner will commit and push; CI then checks lint, database-backed tests, image build, fresh startup/readiness and image export.

Health/readiness and request/provider metrics are implemented. Bounded concurrency, timeouts, a non-root image and shared PostgreSQL retrieval are verified controls. Metrics remain per process; measured production throughput, gateway authentication and per-client rate limits are deployment work. Synthetic measurements do not establish real-customer accuracy. Further dataset expansion is outside the owner's current scope.

## Final Docker response check

[Five owner-reviewed complaints through Docker](../data/evaluation/owner_corrections_verified_docker_20261004.json) returned HTTP 200 with passing citation contracts. All five used local drafting because Groq/Gemini were rate-limited. Four selected the expected procedure; periodic broadband disconnection selected a conditional congestion investigation instead of access-session renewal. This remaining relevance mismatch is open. The final image retains the better-performing fallback-order correction; the title-aware experiment was reverted.
