# Ideas and design journey

This reconstruction uses the request, code and retained experiments. It does not assign undocumented personal reasoning or authorship. Contracts/commands/scores are in [README](../README.md); operational summaries in [design decisions](design-decisions.md).

## Working hypothesis

Retrieve a similar successfully resolved incident and adapt its investigation while requiring current source-backed conditions. Semantic similarity finds paraphrases but does not prove the previous fault is this customer's fault.

The implementation retains KB authority: diagnostic gates, conditional remedies, restrictions and completion. Linked tickets illustrate investigations/outcomes. Unresolved conversations/historical replies are context, not successful repairs. Reviewed recent outcomes fetch current KB references and remain browser-scoped/simulated.

## Considered versus implemented

| Approach | Status | Tradeoff |
|---|---|---|
| Keyword-only/dense-only | Implemented/evaluated baselines | Exact terms versus paraphrases; neither proves diagnostic relevance |
| Hybrid lexical/dense RRF | Implemented | Combines ranks without score-scale calibration |
| BM 25 versus PostgreSQL full-text | Both implemented; Compose uses PostgreSQL | Shared search avoids per-worker corpus copy; compatibility labels need explanation |
| Cross-encoder | Implemented, configurable | Better ordering costs CPU; not telecom-trained |
| Similar-ticket-only solution | Discussed, not adopted safety policy | Old outcomes lack authority for new repairs |
| Open-ended LLM plans | Constrained instead | Protected local gates/steps prevent invented replacement instructions |
| LangChain | Discussed, not integrated | Explicit modules expose validation/fallback without framework dependency |
| Daily answer file | JSON evidence/wording/critique caches implemented | Revision/revalidation matter more than blind answers; no Excel solution store |
| Jev probabilistic classification | Owner considered; not integrated | Owner cited absence of free API; no specific model/provider supplied, so capabilities are not inferred |
| Automatic live-class prediction | Retrieval ingest implemented; prediction separate | New classes require examples, mapping, train/calibrate |

## Changes visible in code

Active KB has ordered checks/gated fixes/completion rather than one engineer note. Prior attempts/latest answers constrain rendering. Applicability differentiates optical loss, weather, daytime slowness, Wi-Fi and settled payments. Primary sequence/conditional alternatives avoid repeated full checklists.

Training compares TF-IDF/MiniLM logistic regression using dev. Calibration controls abstention coverage, not diagnosis certainty. Explicit signals rescue clear complaints; severity/neutral defaults remain estimates without invented quotes.

Providers support split Groq generation/Gemini critique or Groq-to-Gemini generation failover/same-provider critique. Circuits, budgets, masked caches favor immediate local fallback on quota failure. Generated wording is not guaranteed; status stays visible.

## Evidence and future work

Retained comparisons show search strategy/KB filtering affect hits. Final-source retention is lower than retrieval, so selection/applicability need separate quality review. Citations measure traceability, not repairs. Authored challenge ratings remain empty.

Historical short load includes substantial 429/503 rejection. Admission protects resources; total RPS including rejection cannot prove capacity. Later queue/budgets need fresh sustained, varied-query measurements.

Proposed work: provider-reviewed procedures/independent human scoring; language/sarcasm and impact-focused severity evaluation; accepted-request latency under independent controlled load; agent authentication; replica artifact/cache coordination/crash recovery; new-class regression testing alongside independent ingestion. These are future proposals, not completed claims.
