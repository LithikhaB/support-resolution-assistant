# Northstar Telecom synthetic support corpus v1

## Purpose and provenance

This is an AI-authored, deterministic dataset for the telecom support hiring challenge. Northstar Telecom is fictional. The records are not customer incidents, expert-approved operational instructions, or evidence of real-world fixes. No external generation API was called to build the artifacts.

The source catalog `scenarios.psv` contains 60 individually authored diagnostic families across 15 categories. Each family couples reported symptoms with attempted checks, an agent diagnostic finding, a cause, a conditional action, a simulated follow-up and a restriction. Actions are never randomly matched to symptoms.

The dataset is ready for prototype ingestion and baseline experiments. It still needs human review and independently authored evaluation complaints before any claim of realistic performance. Controlled variants must not be counted as independent examples in statistical analysis.

## Artifacts

| File | Content |
|---|---|
| `scenarios.psv` | 60 authored causal scenarios; four per category |
| `tickets.jsonl` | 480 simulated resolved-ticket variants plus 30 unresolved historical responses |
| `knowledge_base.jsonl` | 60 conditional fictional-provider procedures |
| `train.jsonl` | 240 labeled complaints from 30 families |
| `dev.jsonl` | 120 labeled complaints from 15 other families |
| `test.jsonl` | 120 labeled complaints from 15 other families |
| `challenge_queries.jsonl` | 15 manually authored behavioral checks |
| `processed/documents.jsonl` | Searchable corpus: 240 training tickets + 30 unresolved cases + all 60 KB articles |
| `processed/chunks.jsonl` | 330 chunks verified against the current pinned MiniLM tokenizer |
| `quality_report.json` | Counts, automatic checks, source/output SHA-256 hashes and limitations |

Eight variants per family are two authored symptom phrasings crossed with four controlled tones. Severity remains fixed within each family so anger cannot create a critical incident. Development and test tone phrases differ from training phrases. This remains a small, templated sentiment experiment, not a robust sentiment benchmark.

Do not index `tickets.jsonl` wholesale: it contains held-out tickets for audit. Only `processed/documents.jsonl` is the retrieval corpus. Do not feed metadata, reference answers, findings, or category labels into a query classifier; use the `query` field and the intended training targets only.

## Category boundaries

| Label | Scope and distinguishing evidence |
|---|---|
| `broadband_outage` | Complete fixed-line access loss; distinguish optical, session and provisioning faults |
| `intermittent_broadband` | Repeated access interruptions, including Ethernet; distinguish physical loss, power and sessions |
| `slow_broadband` | Throughput problems; compare capable wired devices, timing and local traffic |
| `wifi_connectivity` | Wireless-local issues while the access service may remain healthy |
| `router_ont_hardware` | Power, thermal, port or firmware faults demonstrated by device checks |
| `mobile_coverage` | Radio registration/reception; distinguish location-wide faults from handset faults |
| `voice_call_failure` | Voice-specific failures with other services potentially working |
| `mobile_data` | Packet-data settings, entitlements or allowance; voice may work |
| `sim_esim_activation` | Initial SIM/profile provisioning, device compatibility and authorized verification |
| `number_porting` | Transfer order, authorization and inter-operator routing |
| `sms_otp` | Messaging delivery, sender submission, local filtering and SMS routing |
| `roaming` | Partner, validity, allowance and entitlement while away from the home network |
| `billing_dispute` | Charge correctness; a dispute does not automatically justify a credit |
| `payment_restoration` | Settlement/allocation and service restrictions; payment does not clear every restriction |
| `iptv` | Stream transport, content feed, channel entitlement and local display path |

Ambiguous complaints should permit clarification and multi-intent analysis. These are the supported v1 labels, not a claim that every telecom issue fits one of them. Cancellation is deliberately unsupported in the challenge set; a later approved category/KB update can demonstrate evolving classes without changing the core document schema.

## Outcome semantics

- Resolved synthetic cases have `doc_type=resolved_ticket`, `outcome_status=simulated_resolved`, `metadata.is_synthetic=true`, a scenario-family ID and explicitly simulated outcome evidence.
- The schema rejects synthetic records claiming `verified_resolved`.
- Pending cases remain `historical_response`, with `outcome_status=unknown` and no resolution.
- KB articles have no ticket outcome. Their authority is explicitly fictional-provider policy.
- Source provenance and outcome status must remain visible to future answer generation. Never describe this dataset as verified real support history.

`schema.py` adds the outcome distinction; metadata carries scenario-specific attributes without creating fixed database columns for every category. Migration `004_synthetic_outcomes.sql` prepares existing databases for the extra status without deleting rows. It was tested in a rollback-only test schema, not applied to the active HF database as part of dataset creation.

## Splits and relevance judgments

For each category, two complete scenario families belong to train, one to development, and one to test. All phrasing/tone variants stay with their family. No held-out ticket appears in the searchable corpus. Exact query duplicates are rejected.

All KB procedures remain available, including those applicable to development/test families. This measures retrieval of existing knowledge for an unseen ticket family, not generalization to absent knowledge. KBs do not copy the evaluation complaint verbatim as their body.

Each query has an author-assigned relevant KB and diagnostic/restriction targets. These are provisional relevance judgments; other passages can also be useful. `contrast_candidate_ids` identify same-category alternative diagnoses for review, not automatically trustworthy negative labels. Do not fine-tune a reranker on them until their relevance has been checked.

The 15 challenge cases cover insufficient information, anger versus impact, overlapping classes, already attempted steps, multiple intents, prompt injection, unauthorized requests, settlement ambiguity, out-of-scope queries, unsupported classes, unverified outcomes and misleadingly similar symptoms.

The train examples are self-matches if searched against their own training tickets; do not report those as retrieval evaluation. Use development/test and challenge cases, plus a separately authored blind test set. Freeze test hashes before tuning and report metrics per family/category, not just across repeated variants.

## Reproduce without changing the active corpus

From the repository root:

```powershell
python -m scripts.prepare_synthetic
python -m scripts.chunk_documents --directory data/synthetic/telecom_v1/processed
python -m pytest tests/unit/test_synthetic.py -q -p no:cacheprovider
```

The legacy `python -m scripts.generate_synthetic_data` command now delegates to this builder. It no longer emits randomly paired CSV resolutions. No network API is needed; chunking uses the pinned local tokenizer cache when available.

The active Hugging Face raw files, processed artifacts and PostgreSQL index were retained. Stage the synthetic corpus in a separate database or explicitly isolated schema, validate retrieval there, then switch the application before implementing Day 3's understanding and RAG behavior. Do not append this corpus to the HF index or run the existing indexer over a smaller source set against that index: its retirement checks intentionally reject that operation.

## Review checklist before calling the dataset mature

1. Have a telecom-informed reviewer check diagnostic gates, permitted actions, scope and simulated follow-up criteria.
2. Add independently written complaints with typos, indirect descriptions and realistic missing details; keep them out of training.
3. Review relevance judgments and contrast pairs; include cases with more than one valid evidence source.
4. Measure category errors, clarification/escalation decisions and unsupported recommendations, in addition to retrieval ranking.
5. Keep a small controlled category-update demonstration separate from the frozen evaluation set.
