# Critical implementation and interview notes

## Explain the project in one minute

“We retrieve telecom support evidence even when customers describe the same problem differently. Local embeddings and BM25 provide complementary candidates; reciprocal rank fusion combines their ranks. The next stage will inspect attempted actions and diagnostic prerequisites before drafting cited next steps. Missing evidence should trigger clarification, not a guessed repair.”

Retrieval, local complaint understanding, optional local reranking, evidence diversity and local extractive resolution drafts are implemented. Exact-source citation guards and advisory escalation are implemented. Generative LLM answers and arbitrary-text entailment checks remain planned.

## Dataset choices and limitations

- The assessment permits synthetic data. Our 60 authored diagnostic families support 15 telecom categories with coherent symptoms, findings, actions and simulated outcomes.
- A family has two complaint phrasings crossed with four tones. The resulting 480 resolved variants are not 480 independent incidents. Thirty unresolved cases preserve unknown outcomes.
- Split by family before creating variants: 30 training families, 15 development, 15 test. Tone phrases also differ across splits. Reordering catalog rows must not change artifacts.
- All 60 KBs remain searchable. Holding out tickets does not mean holding out knowledge. Author-assigned relevant KB IDs are provisional, not exhaustive human judgments.
- Never index all of `tickets.jsonl`, train on dev/test, or include diagnostic answers/labels as classifier inputs. Only the train ticket subset and KBs enter `processed/documents.jsonl`.
- `contrast_candidate_ids` identify alternate diagnoses for human review; they are not automatically valid negative labels.
- The corpus is AI-authored, fictional-provider material. Automatic checks establish consistency, not telecom expertise. Independently authored evaluation questions and domain review remain necessary.
- Challenge cases cover ambiguity, multiple intents, out-of-scope requests, unverified outcomes, prompt injection, sensitive codes, prior attempts and anger versus impact. Their expected behaviors are future requirements; retrieval alone does not satisfy them.

## Evidence semantics

`verified_resolved` is reserved for real outcome evidence. Synthetic fixes use `simulated_resolved` with synthetic provenance and family IDs. Historical replies retain `unknown` and cannot carry a resolution. A KB describes a procedure, not a ticket outcome.

Resolutions and agent-only diagnostic findings stay on the parent ticket, outside complaint embeddings. KB procedures contain their diagnostic gates. The generated complaint must not accidentally reveal a diagnosis the customer could not know.

## Design decisions worth defending

- **Local embeddings:** pinned MiniLM, CPU execution, normalized 384-dimensional vectors. No external embedding API. Preserve tokenizer/model revision compatibility.
- **Chunking:** 256-token ceiling including special tokens; overlap supports longer documents. Current 330 documents fit in 330 chunks, but boundary and overlap tests cover longer inputs.
- **Search:** PostgreSQL performs cosine search. Similarity is `1 - cosine_distance`. Unfiltered HNSW is approximate; filtered search ranks all eligible chunks exactly before fetching wide evidence.
- **BM25:** compact in-memory inverted index with positive Robertson IDF, k1=1.5 and b=0.75. Exact tokens help identifiers; lexical similarity does not establish a diagnosis.
- **Fusion:** sum `1 / (60 + rank)` across branches. Raw BM25 and cosine scores are not added. Duplicate candidates must not distort later ranks. RRF is not a probability or confidence score.
- **Lifecycle:** batched writes, fingerprints and checkpoints support resumption. Advisory locks prevent readers from seeing a partly rebuilt corpus. BM25 reloads when index-state fingerprints change.
- **Privacy:** retrieval logs record lengths/counts/timings, not complaint text or database exception details. API errors are sanitized.
- **Readiness:** schema availability, index readiness and model availability are different checks. The first semantic request loads the model; SQL timeout is not an end-to-end request deadline.
- **Scale:** each API process owns its model and BM25 cache. More workers consume more memory. A persistent lexical index, connection pooling and load testing are possible later improvements, not implemented guarantees.

## Implemented local understanding

- Compare train-only TF-IDF against frozen MiniLM features, each with logistic regression at C=1, 4 and 16. Family weights prevent larger families dominating training. Select by development macro-F1; never load test during training.
- Selected MiniLM + logistic regression at C=1: development accuracy 60%, macro-F1 0.552 on 120 variants from 15 families. The encoder is frozen; only the classifier is trained. This small synthetic benchmark does not establish real-world accuracy.
- Porting and payment-restoration development families have zero accuracy; Wi-Fi is also weak. Broader independently authored training scenarios are needed. Preserve the held-out test set while improving data.
- Category scores are uncalibrated. Fixed score/margin gates (0.45/0.10) request clarification on weak predictions; they are not validated out-of-domain detection. Reported accuracy is before these gates, not accuracy among accepted predictions.
- Product mentions, sentiment, impact and prior actions use separate English rules with exact text spans. Negated and hypothetical actions are distinct from completed troubleshooting. Rules are inspectable but cannot understand every paraphrase.
- Severity depends on stated service impact, not anger. Missing evidence remains unknown. Predicted categories are not confirmed diagnoses and are not automatically used as retrieval filters.
- JSON weights avoid arbitrary pickle deserialization. Artifact dimensions, label uniqueness and train/development separation are validated. API errors hide internal details; analysis logs omit complaint text.
- Run `python -m scripts.train_understanding` to rebuild, then restart the API to load new weights. `/api/v1/analyze` needs the local model but no database. The first MiniLM request incurs model-loading latency.

## Implemented reranking and context improvements

- Analysis now retains explicit connection patterns, cable condition and wired-connection results with customer-text spans. These are reported observations, not independently verified diagnoses or proof that a cable test was performed.
- Contact and next-step requests are separate from the technical category. Clarification uses known service details and asks for provider/region for contact lookup. Contact lookup is still pending. The local draft layer now presents conditional KB procedures for agent review.
- The pretrained `cross-encoder/ms-marco-MiniLM-L6-v2` jointly scores each query/chunk pair on CPU. Its revision is pinned; no hosted API or reranker fine-tuning is used.
- Reranking is opt-in, defaults to 20 candidates and caps the pool at 50. It cannot recover evidence absent from the retrieved pool. Scores are raw relevance logits; they are neither calibrated confidence nor diagnostic proof.
- Original source ranks, content and outcome status are preserved. Long pair truncation is explicit. Model failure returns an error; it does not masquerade as successful reranking.
- `scripts.evaluate_reranking` compares identical candidate pools on development queries with authored KB relevance. The report includes top-five hit rate, reciprocal rank, candidate coverage and local timing. Repeated variants are correlated; these are development measurements only.
- Measured reranking regression: expected-KB hit@5 dropped from 50.0% to 36.7%, with MRR@5 dropping from 0.461 to 0.341. Candidate hit rate at 20 was 62.5%; median additional ranking time was about 759 ms. Keep reranking optional. Synthetic sibling tickets can crowd out KBs, and KB-only authored labels may omit other useful evidence; review both diversity and relevance judgments before drawing broader conclusions.
- The classifier and its 60% development accuracy are unchanged. Better extraction does not establish improved classifier accuracy.

## Remaining local model work

1. Improve category generalization with reviewed independent scenarios; calibrate routing and evaluate accepted coverage before relying on automatic routing.
2. Expand and independently evaluate the implemented action and impact rules. Few independent examples support critical-impact learning here.
3. Review reranker failures and duplicate scenario variants before considering fine-tuning. Candidate coverage and evidence diversity can limit the result even when pairwise scores improve.
4. Retain the embedding baseline initially. Fine-tune only with reviewed positive/negative pairs and demonstrated held-out improvement.
5. Use one replaceable final-drafting model. Groq may serve it; classification, embeddings, retrieval, reranking and validation need not call Groq. A local quantized Qwen3-4B is an experiment contingent on hardware and measured latency, not a current dependency.

Official references: [MiniLM embeddings](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [cross-encoder reranker](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2), [logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression), [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B).

## What would make the demonstration distinctive

- Compare complaints differing only in attempted troubleshooting; demonstrate why the next action changes.
- Contrast superficially similar faults: Wi-Fi versus wired outages, pending holds versus settled duplicate payments, HDMI failure versus buffering.
- Ask for a missing diagnostic observation instead of inventing a cause.
- Reject nonexistent citations and unsupported actions. Citation existence alone is not proof of factual support.
- Show an unsupported category, introduce approved knowledge and labels, then demonstrate the update without changing storage. Keep this separate from the frozen evaluation set.
- Report keyword, vector, hybrid and reranked ablations, latency and failure examples. These are engineering differentiators, not claims of research novelty or guaranteed hiring success.

## Operations and rollback

The development application now uses database `support_telecom` and `CORPUS_DIR=data/synthetic/telecom_v1`. The old HF database `support_db` and ignored raw/processed files remain for rollback; its download/normalization code is retired.

Restart any running API process after the switch to reload settings and caches. Never append synthetic documents to the old HF index. The indexer rejects silent removal of existing documents or model-profile changes; those require an explicit replacement workflow. Never delete a Docker volume as a routine migration step.

Do not expose this local prototype publicly as a completed production deployment. Authentication, rate limiting, full request deadlines, operational load testing and arbitrary generated-answer validation are not yet implemented. The current extractive draft has exact-source integrity checks. Do not add them merely to inflate scope; complete and measure the intended next phase first.

## Diverse retrieval and conditional resolution drafting

- `--diversify` on `scripts.test_retrieval` overfetches at least 50 candidates, keeps one chunk per document and collapses synthetic siblings by provider/source, dataset version, scenario family and document type. Real cases are not collapsed merely because they share a family label. KBs and tickets remain separate evidence types. Ranking scores are unchanged.
- On all 120 development queries (15 families), expected-KB hit@5 increased from 50.0% to 61.7% with diversity. KB-focused retrieval reached 80.0%; corresponding MRR@5 values were 0.461, 0.495 and 0.694. KB filtering narrows the task to finding procedures; it is not a universal search improvement or a measure of correct diagnoses. Labels are provisional and variants are correlated. Test was untouched at that stage; the frozen pipeline evaluation below records its later measurement. See `data/evaluation/diversity_development.json`.
- `POST /api/v1/resolve` and `python -m scripts.resolve_complaint "..."` combine local understanding, diverse KB-focused search and an extractive draft. No new model is trained and no hosted LLM is called. Reranking was disabled at that milestone; the context-refinement phase below enables it by default for resolution drafts only.
- The draft copies diagnostic gates, conditional actions and restrictions from complete fictional-provider KB chunks. Each copied field has an exact source span and a citation pointing to its document and chunk. Unknown formats, incomplete chunks and wrong provenance are excluded; this version does not interpret arbitrary uploaded prose.
- Explicit product mentions constrain which retrieved procedures can appear in a draft; the uncertain classifier category never becomes an automatic intent filter. Product mentions alone do not prove relevance, so every procedure remains a possibility requiring agent review. Difficult examples can still retrieve unsuitable alternatives.
- Reported completed actions are acknowledged. If a proposed procedure repeats a recognized action, the draft withholds that repeat for review. Negated attempts do not count as completed actions. Recognition is rule-based and limited to the current complaint; there is no persistent conversation memory yet.
- Customer statements never automatically satisfy provider diagnostic gates. This phase produces conditional suggestions, not an executed repair or a confirmed diagnosis. Fictional-source provenance remains visible. Contact requests produce no invented phone number.
- Status distinguishes missing service details, insufficient applicable evidence and a draft requiring diagnostic confirmation. Infrastructure failures return sanitized 503/504 errors instead of being represented as a lack of evidence.
- Verified: 218 tests passed, including diversity, repeat suppression, negation, incomplete/duplicate source fields, exact citation spans, input contracts and error handling. Real-model CLI and API checks exercised the broadband example, absent evidence and unsupported service text. That was the earlier validation milestone. The next section records the citation guard and evaluation additions; diagnosis-quality review, external ticket handoff and UI remain later work.

## Citation guards, workflow decisions and frozen evaluation

- Every resolution response now includes `validation` and `decision`. The guard checks retrieved document/chunk identity, exact source spans, provenance, scope, diagnostic conditions, proposed actions, restrictions, previous-action state and deterministic rendered text. Invented actions, omitted restrictions, orphan citations and altered prose fail validation. All proposed procedures are withheld on failure.
- These checks establish exact support for the extractive contract. They do not establish that a retrieved procedure applies to the customer, that the fictional procedure is correct, or that arbitrary generated prose is logically entailed. A 100% citation-contract pass rate must never be called 100% resolution accuracy.
- The workflow recommends clarification, agent review or escalation with explicit reasons and priority. A recognized area outage receives urgent network-operations review; anger alone does not increase urgency. Missing applicable evidence, failed citation checks and exhausted recognized actions recommend agent review through escalation. A support-contact request remains unverified until provider contact information exists.
- Escalation is advisory: `handoff_created=false`. The application does not contact anyone or create external tickets. Recognition depends on the implemented English rules; unrecognized impact can remain unknown.
- `python -m scripts.evaluate_pipeline --split dev --output data/evaluation/pipeline_dev_repeat.json` runs two classifier comparisons and six retrieval configurations. Existing report paths cannot be overwritten. Each run freezes code, model, data and settings hashes before loading evaluation queries, checks family/text separation and verifies the live index matches the frozen corpus. It rechecks fingerprints before publishing.
- The TF-IDF comparison is refit on training rows using the C selected from the existing development report. The published MiniLM classifier is loaded unchanged. Test labels select neither model nor thresholds. Test results must not be used for tuning; a later rerun is not a new unseen holdout.
- Development results: selected MiniLM accuracy 60.0%, TF-IDF 37.5%; neither classifier accepted any of 120 queries under the existing fixed score/margin gates. Severity was unknown in 90.0% of cases, with 3.3% agreement against the authored severity labels. Tone agreement was 75.0%. These are priority gaps for data/rule review and development-only calibration.
- Development KB hit@5: BM25 50.8%, vector 50.8%, hybrid 50.0%, diverse hybrid 61.7%, KB-focused 80.0%, KB-focused plus reranking 90.0%. KB-only reranking is a different task from the earlier mixed-ticket reranking regression. Runtime defaults remain unchanged. Default drafts contained the expected KB in 60.0% of development cases.
- The 15 challenge outputs are retained for manual behavioral review. Exact citation checks passed, but generic clarification does not satisfy every expected challenge behavior: explicit decline/abstention, issue separation and certain specialized routing behaviors still need work. Do not present the challenge set as fully passed.
- Verified implementation checks: 246 tests passed; lint and formatting passed. Live API cases confirmed normal clarification, urgent area-outage escalation and normal priority for an angry billing complaint with working service. README.md was left untouched as requested.

## Frozen held-out results and remaining priorities

The completed `data/evaluation/pipeline_test.json` records 120 held-out variants from 15 scenario families. `pipeline_test.freeze.json` identifies the code, model, corpus and settings fixed before loading those queries. The interrupted evaluation was resumed with identical fingerprints; runtime defaults were not changed in response to results.

| Measurement | Held-out result |
|---|---:|
| Published MiniLM category top-one accuracy / macro-F1 | 74.2% / 0.713 |
| Development-selected TF-IDF accuracy / macro-F1 | 40.8% / 0.364 |
| Category acceptance at existing score/margin thresholds | 0 of 120 for both classifiers |
| BM25 expected-KB hit@5 / MRR@5 | 39.2% / 0.320 |
| Vector expected-KB hit@5 / MRR@5 | 54.2% / 0.458 |
| Hybrid expected-KB hit@5 / MRR@5 | 56.7% / 0.429 |
| Diverse hybrid expected-KB hit@5 / MRR@5 | 72.5% / 0.487 |
| KB-focused expected-KB hit@5 / MRR@5 | 94.2% / 0.798 |
| KB-focused plus reranking expected-KB hit@5 / MRR@5 | 100.0% / 0.872 |
| Default drafts containing the expected KB | 57.5% |
| Drafts containing any selected sources | 84 of 120 |
| Citation contract pass rate | 100.0%, including zero-source responses |
| Severity agreement / unknown rate | 10.0% / 90.0% |
| Tone agreement | 100.0% on these templated synthetic tones |

All 120 ordinary held-out cases requested clarification under the current workflow. The separate live outage example and challenge case exercise urgent escalation. Successful rule examples do not establish broad impact-detection coverage.

The retrieval scores are for author-assigned KB labels, and variants within a family are correlated. KB-only retrieval excludes competing ticket evidence. Neither the 100% reranked hit rate nor the citation-contract pass rate means 100% correct resolutions. At that frozen milestone, the default draft used KB-focused hybrid retrieval without reranking, and the reranked comparison was an ablation. Later context-refinement behavior is recorded separately below.

Priority work before a polished UI: broaden service/impact recognition, calibrate abstention on development data, and review why relevant retrieved evidence fails to reach the final draft. Multi-issue separation, explicit unsupported-request handling and specialist routing also remain incomplete. Keep these test results as the frozen baseline; use a newly authored independent evaluation set for future unbiased claims after iterative improvements.

To demonstrate the completed guard and decision features, run `python -m scripts.resolve_complaint "Our street and neighbouring blocks all lost broadband at once." --json`. Inspect `validation`, `decision`, `suggestions` and `sources`. Ordinary CLI output also prints validation status and the advisory next action. No new model download or retraining is needed.

## Context refinements after reviewing the pasted output

- The pasted shared-outage response escalated correctly but suggested household Wi-Fi repairs. Exact citation copying was insufficient to establish contextual relevance. The incident path now asks for area and start time, augments retrieval with reported regional scope, and only permits procedures whose diagnostic gate describes an area/major incident. The citation guard independently applies this exclusion. If no applicable procedure is available, escalation remains urgent without local repair guesses.
- Explicit shared broadband loss can produce `category=broadband_outage` with `category_basis=explicit_report` and supporting spans. Original model candidates remain visible. This is a service-impact category, not proof of a root cause or a claim that the classifier predicted correctly.
- Service mentions and impact rules recognize additional wording such as calls failing, a suspended line, router shutdown, text-message problems and wired devices. Negated and hypothetical area failures must not trigger urgent escalation. Rules still do not cover arbitrary language.
- `python -m scripts.calibrate_routing` selects a model-bound profile using development data only: maximize acceptance while retaining at least 85% accepted-example and accepted-family mean accuracy, at least five accepted families and 20 accepted examples. The chosen raw-score threshold is 0.34 and margin is 0.15: 28/120 examples across seven families, with 85.7% accepted accuracy. Compatible service mentions are required. These are optimistic development-selection measurements because the same development split selected the classifier; they are not independent accuracy or calibrated probabilities.
- Missing, malformed or model-mismatched routing profiles fall back to fixed conservative thresholds. Retraining should be followed by recalibration and API restart. No classifier weights were retrained in this refinement.
- Resolution requests now use the already cached KB reranker by default; ordinary `/retrieve` defaults are unchanged. `scripts.resolve_complaint --no-rerank` preserves an explicit comparison path. This choice follows the earlier development KB-only comparison, not the old held-out result.
- A draft whose next action is clarification now presents the missing checks first and withholds speculative repair instructions from the prose. Conditional procedures remain in JSON for agent review. Scope exclusions are narrow and do not establish general diagnostic entailment.
- Each new CLI invocation loads the local models into a new process. An already-running API reuses them, so the first request and repeated requests have different latency. No hosted API key is required.
- `scripts.evaluate_refinements` measures the actual current resolution service on development queries only. The old test reports remain unchanged; no new unseen-test claim is made after reviewing the earlier held-out results.
- The completed refinement run accepted 32/120 development categories (28 model-based and four explicit-report cases), with 87.5% accuracy among accepted cases. Drafts contained the expected KB in 85% of cases and any sources in 116/120. All citation-contract checks passed, including responses without sources; this does not measure resolution correctness. Decisions were 108 clarification, eight agent review and four escalation.
- Severity remains a substantial limitation: agreement was 26.7%, with 63.3% unknown. Category abstention also remains frequent (88/120). Broader independently authored language and impact evaluation are needed before claiming dependable automation.
- Validation completed with 268 passing tests, lint and formatting checks, and live API assertions for both original complaints. The shared outage selected the regional incident KB and urgent escalation without household repair advice; intermittent broadband acknowledged the prior restart and asked for clarification before repairs. Results are saved in `data/evaluation/context_refinement_development_v2.json` and `data/evaluation/context_refinement_smoke_v2.json`.

## Complaint handling and follow-up conversations

- Complaint analysis now reports `scope_status` and `scope_reason`. Recognizable unrelated requests such as asking for a poem receive an explicit unsupported response without retrieval or repair advice. Unrecognized language remains uncertain; these narrow English rules are not a general out-of-domain classifier.
- Impact rules cover more explicit complete-loss and degradation phrases, exclude questions and hypothetical incidents, and prioritize complete loss over degradation. No new accuracy percentage is claimed; prior frozen evaluations remain historical results.
- `POST /api/v1/conversation` separates explicit changes of service into at most four issues. Broadband, router and Wi-Fi belong to one connectivity issue. Each issue has its own draft, evidence, questions and customer replies. This does not reliably split every compound complaint or multiple problems within the same service.
- Conversations are stateless: send the unchanged original `query` and full `turns` list on every request. The server stores no conversation history. Each turn requires an `issue_id` and a `message`, `observations`, or both. Up to eight turns are allowed; character and model-token budgets still apply. Over-budget input is rejected rather than silently truncated.
- Recognized later observations replace earlier observations of the same kind. Conflicting facts within one text turn remain visible for clarification. Structured answers take precedence within the same turn. Earlier troubleshooting attempts remain acknowledged and are not automatically recommended again.
- Supported structured observations include `wired_connection` (working/failing), `wireless_devices` (one/all), `impact` (complete_loss/intermittent/working), `mobile_services` (calls/texts/data/several), `billing_status` (pending/settled), `charge`, `tv_symptom` (no_picture/error/buffering), `area`, `started`, `provider` and `region`. Service-specific fields must match the chosen issue. Providing a provider name does not create a verified helpline.
- Supplying the missing Ethernet observation advances the Wi-Fi question. Supplying the wireless-device observation removes that answered question. Unknown answers do not silently mark a question answered. These are customer observations, never authorization or confirmation of a provider diagnostic gate.
- A customer report that service is working withholds further repair instructions for agent outcome review. A previously reported shared outage retains urgent incident review even when one customer reports recovery. This is not automatic ticket closure or a real external handoff.
- Each issue returns `analysis_text`, the normalized text to which its evidence offsets refer. Structured fields are represented as explicit customer-report statements in that text. The frontend should use these returned offsets, not offsets into the original unsplit complaint.
- Validation: 305 unit and rollback-only PostgreSQL integration tests passed, with lint and formatting checks. Tests cover issue isolation, retained attempts, correction order, same-turn contradictions, recovery, unsupported scope, invalid history, wrong-service observations and question progression. Existing dependency deprecation warnings do not indicate a failed request.
- The original shared-outage CLI output was parsed as valid JSON. Live local-model and database checks exercised initial complaint, Ethernet answer, wireless-device answer, two-issue separation and unsupported requests. Results are saved in `data/evaluation/conversation_smoke.json`; these selected regression examples are not an independent accuracy evaluation.

Run a follow-up example with `python -m scripts.converse --query "My broadband drops. I already restarted the router twice." --reply "Ethernet works." --reply "All wireless devices disconnect."`. Add `--json` for full evidence and per-issue analysis. For multiple issues use `python -m scripts.converse --query "My broadband drops. Also my bill has a duplicate charge."`. `--issue-id 2` targets replies to the second issue. A structured request can be loaded with `--request path-to-request.json`.

In Swagger, use `/api/v1/conversation` with this body, retaining earlier turns when adding another answer:

```json
{
  "query": "My broadband drops. I already restarted the router twice.",
  "turns": [
    {
      "issue_id": 1,
      "message": "Ethernet works.",
      "observations": {"wired_connection": "working"}
    },
    {
      "issue_id": 1,
      "observations": {"wireless_devices": "all", "impact": "intermittent"}
    }
  ]
}
```

## Local launch and agent workspace verification

- Start from the project folder with `./run.ps1`. It explicitly runs `.venv/Scripts/python.exe`, so an inactive environment cannot accidentally select global Python. `./run.ps1 -Check` verifies imports without starting a server. The equivalent direct command is `.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload`.
- `ModuleNotFoundError: No module named 'psycopg'` in the global Python installation was an interpreter-selection problem. The existing project environment successfully imports psycopg and the application. If rebuilding that environment, install the full `requirements.txt` through its Python executable.
- The browser workspace is at `http://127.0.0.1:8000/`; Swagger remains at `/docs`. Cases and reviews persist in `data/workflow/cases.sqlite3`, configurable through `CASE_STORE_PATH`. This is a local single-user demonstration without authenticated agent identities. Keep real customer data out of demonstration cases.
- Browser verification covered case creation, accepting a draft, editing with review notes, recording an unresolved outcome with evidence, reloading saved edits, rejecting a draft, expanding review history, and downloading a handoff JSON file. The handoff retained `delivery=local_draft_only`, `handoff_created=false` and agent-edit provenance. A 390-pixel layout was visually checked; controls stack vertically.
- Reviews are tied to a generated response version. New customer replies invalidate the applicability of earlier reviews while retaining their original resolution snapshots. Agent edits do not inherit automated citation validation. Reported resolution outcomes are agent-entered observations, not verified repairs or training labels.
- `python -m scripts.evaluate_acceptance` runs eight authored behavioural regression cases and preserves an inspectable report. The recorded run passed 8/8, with a 33.7-second first request and approximately 1.5-second median for subsequent requests on this machine. These mixed cases are not a controlled load benchmark or independent held-out accuracy measurement. Use a new `--output` path for another run; previous reports are not overwritten.
- Final regression verification: 324 tests passed, including rollback-only PostgreSQL integration tests; lint and formatting checks passed. Dependency deprecation warnings remain. Fresh independently authored evaluation, a clean-environment setup rehearsal and final submission documentation remain before making final quality claims.
