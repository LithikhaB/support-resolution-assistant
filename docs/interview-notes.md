# Critical implementation and interview notes

## Explain the project in one minute

“We retrieve telecom support evidence even when customers describe the same problem differently. Local embeddings and BM25 provide complementary candidates; reciprocal rank fusion combines their ranks. The next stage will inspect attempted actions and diagnostic prerequisites before drafting cited next steps. Missing evidence should trigger clarification, not a guessed repair.”

Retrieval, local complaint understanding and optional local reranking are implemented. Answer generation and citation-support checks remain planned.

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
- Contact and next-step requests are separate from the technical category. Clarification uses known service details and asks for provider/region for contact lookup. Contact lookup and generated troubleshooting are still pending.
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

Do not expose this local prototype publicly as a completed production deployment. Authentication, rate limiting, full request deadlines, operational load testing and final-answer validation are not yet implemented. Do not add them merely to inflate scope; complete and measure the intended next phase first.
