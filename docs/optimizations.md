# Conversation memory and guarded reuse

The active corpus is `telecom_v3_1`: 62 AI-authored synthetic KB procedures,
240 simulated training-ticket resolutions and 30 historical responses (332 indexed
records). V3 remains unchanged. Two additional baseline investigations cover an
outage without an established fault and persistent slowness without peak-hour
evidence. Their five steps collect observations, check provider diagnostics,
conditionally route a confirmed fault and verify recovery. They have no invented
resolved cases attached. No classifier retraining was needed for this update.
Development and test split files are byte-identical to V3 and excluded from indexing.

## Request flow

1. Validate and scrub credentials from the complaint and bounded issue-specific
   follow-ups. Reject recognized non-telecom requests without generating guidance.
2. Combine the local category classifier with exact quoted product, impact,
   sentiment, completed-action and observation rules. The demonstrated Tanglish
   phrases and sarcasm are narrow authored rules, not general multilingual competence.
3. Search PostgreSQL pgvector and full-text ranks, fuse with RRF and optionally
   rerank. Exclude procedures that conflict with reported observations. Vague
   outage/slowness complaints use baseline investigations instead of assuming a
   migration, speed upgrade or congestion diagnosis.
4. Retrieve training histories and the current browser's explicitly reviewed
   histories. A history must link to a selected KB procedure; historical step
   citations require exact matching source steps. Similarity alone is not proof.
5. Render ordered, cited steps, preserving conditional diagnostic gates, completed
   actions and restrictions. Validate source spans and citations. Optional provider
   wording cannot replace V3's ordered local steps; failure retains the local plan.
6. Return advisory decision, priority, focused question, source provenance and
   provider/fallback labels. Save a redacted conversation when enabled.

KB remains necessary for approved conditions and restrictions. An unresolved
conversation never becomes a repair authority, and an old customer's successful
repair never confirms the cause of a new customer's complaint.

## Persistence and review

Enable `CONVERSATION_STORAGE_ENABLED=true`, run database setup, and use the UI's
Recent conversations button. PostgreSQL stores redacted request snapshots and
validated responses. The browser keeps only a conversation UUID in localStorage;
an HTTP-only SameSite cookie grants access to that browser's records. Refreshing
regenerates the current response from saved input rather than displaying a stale
plan. Follow-ups carry a revision: stale revisions return HTTP 409.

Records expire after `CONVERSATION_RETENTION_DAYS` (default seven). Expired owned
records are removed during storage access; there is no background cleanup daemon.
The UI's Delete action cascades through snapshots and reviewed histories. There is
a bound of 100 conversations per browser and 20 snapshots per conversation.

To add reviewed history, expand the issue's reviewed-outcome form, describe the
observed result (at least 20 characters), check the explicit confirmation and save.
The server obtains the resolution from its validated snapshot, not submitted repair
instructions. Every saved demonstration outcome remains `simulated_resolved` and
is eligible only for that browser and the same published corpus revision. This is
operator acknowledgement, not expert verification or proof of a real repair.

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/conversation` | Resolve and save redacted input; accepts conversation_id and revision |
| `GET /api/v1/conversations` | List recent records owned by the browser |
| `GET /api/v1/conversations/{id}` | Load saved input and revision |
| `DELETE /api/v1/conversations/{id}` | Delete owned conversation and associated histories |
| `POST /api/v1/conversations/{id}/review` | Explicitly acknowledge a simulated reviewed outcome |

Review requires `issue_id`, current `revision`, `outcome_note` and literal
`confirmation: "simulated_resolution_reviewed"`. These APIs reject another
browser's IDs. They are demo browser capabilities, **not authenticated agent or
customer identities**. Production shared history needs authenticated tenant access,
audited review and retention controls. Clearing the cookie removes browser access;
it does not authenticate a different customer.

## Cache safeguards and limits

- Query embeddings: bounded in-process TTL cache, keyed by model identity and
  query digest. Stored values are vectors, not raw query strings.
- Retrieval: bounded in-process TTL cache keyed by published index revision and
  the entire search request, including filters, depth and rerank options. Cached
  hits still check database readiness and indexing locks. Failed searches are not cached.
- Semantic reuse: browser-owned selected evidence can be reused above cosine
  similarity 0.98 only when category, products, severity, sentiment, observations,
  full attempted-action text, retrieval options, provider settings and corpus
  revision match. It reuses candidate evidence, not a previous generated answer.
  Fresh drafting, applicability, policy, history lookup and exact-source validation
  still run. The UI labels a revalidated semantic hit.

`COMPUTATION_CACHE_SECONDS` defaults to 120; zero disables expiry-based reuse.
Caches are bounded to 128/256 entries, local to one process and cleared on restart.
They are not a Redis deployment or a measured production throughput guarantee.
The conservative semantic threshold often misses paraphrases intentionally.
Changing observations prevents unsafe reuse. Reviewed histories are fetched fresh.

Live provider quotas are unchanged; local regression verification does not
establish provider-generated answer quality. The deterministic plan is the reliable
demo path, with human review required before any action.
