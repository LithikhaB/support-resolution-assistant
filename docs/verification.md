# Verification — 5 October 2026

The active Docker application uses `telecom_v3_1` on port 8000 with PostgreSQL/pgvector on port 5432. The corpus has 332 documents: 62 synthetic procedures, 240 simulated resolved training tickets and 30 historical responses. Dev/test tickets are not indexed.

## Reproducible checks

```powershell
python -m ruff check
python -m ruff format --check
$env:RUN_DB_TESTS='1'
python -m pytest -q
node --test tests/web/agent_console.test.cjs
python -m scripts demo --output data/evaluation/my_new_http_run.json
```

Latest full local run: 50 Python tests passed with DB tests enabled; five console tests passed. Dependency deprecation warnings remain. Tests cover shared quota budgets, Retry-After cooldown, JSON-cache expiry/restart reuse, private history ownership, revision invalidation and protected LLM output.

## Recorded evidence

- `data/evaluation/submission_challenge_release_20261005.json`: 53 cases, 66 snapshots, local mode. All cases pass the citation-validation/agent-review contract checks. Per-area summary: `cache_roles_challenge_summary_20261005.json`: follow-up 10/10, working-vs-failing 9/9, safety 11/11, natural wording 9/9, scope 11/11, retrieval contrast 3/3. Manual semantic scores remain unfilled; these are not accuracy scores.
- `submission_live_verified_20261005.json`: two accepted Gemini-generated plans under the earlier failover configuration. Other live runs include rejected output, timeout and quota fallbacks. This does not verify the later split-provider configuration.
- `submission_evolution_final_20261005.json`: all seven new-category evolution checks passed in a separate database; active corpus/model fingerprints stayed unchanged.
- `v31_postgres_dev_release_20261005.json` and `v31_postgres_test_release_20261005.json`: frozen PostgreSQL lexical retrieval ablations, with accompanying freeze manifests. Reranked KB hit rates: 0.9333 / 1.0; expected final KB retention: 0.825 / 0.80. These precede the latest narrow context fixes and are not rerun/tuned test results. The compatibility label `bm25` means PostgreSQL full-text rank for these runs.
- New HTTP runs use `cache_roles_*_http_20261005.json`. The first was issued during startup and records transport failures; the next exposed an unrelated payment procedure. Neither failed report is overwritten.

## Limits and deployment status

All content is AI-authored synthetic. Simulated outcomes stay `simulated_resolved`. Exact citation checks and model critique do not establish real-world correctness. General paraphrases, sarcasm and multilingual input are not fully covered. An agent must review the proposed plan and verify diagnostic gates.

The cache is a bounded local JSON implementation suitable for the single application container. Rate limits/cooldowns use shared PostgreSQL budgets, DB connections are pooled, and expensive request concurrency is bounded. Multi-host cache sharing, authenticated agent/customer identity and production load/SLA verification remain deployment work. Browser ownership is a capability cookie, not an identity system.

These working-tree changes have not been pushed; remote CI has not verified this revision. Local checks must not be described as a successful GitHub Actions run.
