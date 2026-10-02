# Local models and interview demonstration plan

This is a proposed next implementation sequence, not a claim that these models have been trained or that these features already work.

## What to train

1. **Category classification:** train a TF-IDF + logistic-regression baseline on `train.jsonl`, using only complaint text. Compare it with frozen local MiniLM embeddings + logistic regression on identical family-separated splits. Fit TF-IDF on train only. Select thresholds/settings on development only; report a frozen test confusion matrix and macro-F1. There are only 30 independent training families, so prefer a small baseline before transformer fine-tuning. Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
2. **Severity and attempted steps:** start with explicit, inspectable impact rules and text-span extraction. Customer anger must not set priority. Critical regional impact is rare here, so the dataset does not support a credible fully learned severity classifier yet. Test calm-critical and angry-low counterfactuals separately.
3. **Sentiment:** start with a baseline, but do not cite high accuracy on generated suffixes as real capability. Expand independently worded, reviewed sentiment examples before choosing a trained model.
4. **Embeddings:** retain the current local `sentence-transformers/all-MiniLM-L6-v2` baseline and pinned revision. No immediate retraining. Fine-tune only if reviewed contrast pairs and held-out evaluation demonstrate a need. Reference: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
5. **Reranking:** start with local `cross-encoder/ms-marco-MiniLM-L6-v2`, scoring a bounded candidate set after hybrid retrieval. It is pretrained on passage ranking, not this telecom domain; benchmark the actual latency and improvement here. Fine-tuning is a later experiment, not a prerequisite. Reference: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2
6. **Final drafting:** one optional hosted generation call using selected evidence, or benchmark a local quantized Qwen3-4B candidate if hardware permits. Do not promise CPU latency without measurement. Make the provider replaceable. No generation-model training is justified by this small synthetic corpus. Reference: https://huggingface.co/Qwen/Qwen3-4B

Groq is a serving provider, not the task-specific model we are training. The current embeddings, database search, BM25 and rank fusion already run without Groq. Classification, reranking, evidence selection and validation can also remain local; generation is the only planned step that may need a hosted model.

## Distinctive, testable behaviors

- **Attempt-aware recommendations:** identify what the customer already tried; do not repeat it as the main action unless new evidence justifies repetition. Test with otherwise-identical complaints that differ in attempted steps.
- **Diagnostic prerequisites:** similar symptoms can have different causes. Match the available findings to each KB's prerequisites; request a missing observation instead of confidently selecting a repair. Test Wi-Fi versus fixed-line loss, pending versus settled payments, and HDMI versus stream faults.
- **Evidence and outcome provenance:** distinguish simulated resolution, unresolved reply and KB guidance. Validate citation existence and source coverage. Citation existence alone does not prove factual support.
- **Abstention and new categories:** route unsupported complaints; then introduce an approved category/KB update and show behavior before and after without changing the storage schema. Avoid using this demonstration to tune the frozen test set.
- **Measured improvements:** compare keyword-only, vector-only, hybrid and hybrid-plus-reranker retrieval on the same reviewed cases. Report latency as well as relevance. RRF scores are not calibrated confidence.

These are engineering differentiators to implement and measure, not claims of research novelty or a guarantee of a hiring outcome. A defensible interview story includes a failure example, what was measured, and why a simpler baseline was retained or replaced.

## Next steps

1. Review the staged synthetic corpus and relevance targets; provision a separate synthetic retrieval database and run its retrieval smoke checks.
2. Switch the development application to that validated corpus before building Day 3. Retain the HF corpus as a separate engineering experiment and rollback source.
3. Train/evaluate the local classifier baselines; implement impact and attempted-step extraction.
4. Add local reranking and evidence selection, then a replaceable final-draft model and citation/unsupported-claim checks.
5. Run ablations on a frozen, independently reviewed test set. Do not select a model based on its own generated training examples.
