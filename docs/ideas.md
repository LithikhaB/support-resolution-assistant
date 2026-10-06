# Ideas and Design Choices

## Starting Idea

- **Find** a similar resolved complaint, **learn** from its investigation, and give the agent a **clear plan with sources**.
- **Core challenge:** keep the plan **relevant** without assuming that two similar complaints share the **same cause**.

---

## Approaches Explored

### 1. Jev for probabilistic categorization
- **Why:** another way to classify complaints.
- **Decision:** **not integrated**, because no **free API** was available.

### 2. TF-IDF + logistic regression
- **Why:** a **simple, cheap, inspectable** local baseline.
- **Decision:** **rejected as the main model.** It struggled with **unseen complaint families**: dev accuracy **56.67%** vs **76.67%** for the published MiniLM classifier.

### 3. LLM-assisted understanding
- **Why:** interpret **varied wording** and extract context beyond fixed keyword rules.
- **Decision:** added as an **optional, quote-validated** extractor.
  - It **supplements** the local model and never replaces it.
  - On **quota failure**, it falls back to **local rules**.

### 4. MiniLM + logistic regression + explicit signals (**chosen**)
- **Why:** **semantic features** handle paraphrases, while **rules** preserve clear observations (optical loss, work impact, completed restarts).
- **Decision:** this is the **current local path**.
  - It can **abstain** and **ask a question** instead of forcing a category.

---

## Current Pipeline

### Understanding
- **Choice:** MiniLM category classifier, quoted rules, optional LLM extraction.
- **Why:** captures **paraphrases** while keeping category, impact, emotion and previous actions **explainable**.
- **Tradeoff:** language variation can still be **misread**, and defaults are **estimates**.

### Search
- **Choice:** **dense vectors** + **PostgreSQL full-text search**.
- **Why:** semantic search finds **similar wording**; lexical search keeps **exact terms** such as *LOS* and *settled payments*.
- **Tradeoff:** **neither channel alone** guarantees relevance.

### Retrieve
- **Choice:** **RRF**, **diversity** and **applicability filtering**.
- **Why:** merges ranks, cuts **repeated evidence**, and rejects procedures that conflict with the reported **service, timing or symptoms**.
- **Tradeoff:** filters can **discard useful evidence**, so source retention needs evaluation.

### Re-rank
- **Choice:** **optional cross-encoder**.
- **Why:** scores the **complaint and candidate together** before a procedure is chosen.
- **Tradeoff:** better ordering costs **CPU and latency**.

### Build the plan
- **Choice:** **ordered KB checks** + **linked resolved tickets**.
- **Why:** the KB defines **diagnostic conditions and restrictions**; tickets show **how an investigation was resolved** in the simulation.
- **Tradeoff:** a historical outcome is an **example, not proof** of the current cause.

### Improve wording
- **Choice:** **Groq** generates, **Gemini** critiques (split mode).
- **Why:** clearer, editable instructions while **protected gates, remedies and citations** stay intact.
- **Tradeoff:** needs **keys and quota**; rejected wording **falls back locally**.

### Validate and decide
- **Choice:** **exact-source checks** + **advisory policy**.
- **Why:** claims stay **traceable**; the result is **clarify, agent review or escalate**, and **no repair is executed**.
- **Tradeoff:** citation validity **does not prove** diagnostic correctness.

### Reuse context
- **Choice:** **revision-aware caches** + **reviewed recent outcomes**.
- **Why:** avoids repeated work, and **unresolved conversations** are never treated as successful solutions.
- **Tradeoff:** reused items must still pass **current applicability and validation**.

### Evolve knowledge
- **Choice:** **authenticated live ingest** + **incremental indexing**.
- **Why:** new KB is **searchable without a restart**, and **unchanged documents** need no new embeddings.
- **Tradeoff:** new classifier categories still need **examples, mapping, training and calibration**.

---

## Storage and Deployment Decisions

### PostgreSQL + pgvector (instead of a separate Qdrant service)
- **Reason:** vectors, document metadata, conversations and **shared budgets** live in **one database**, so there are **fewer services** to run and review.
- **Boundary:** a **simplicity** decision, **not a benchmark** showing pgvector beats Qdrant. Larger vector workloads need **separate measurement**.

### One modular FastAPI microservice
- **Reason:** understanding, retrieval, drafting and ingestion stay **separate in code** without splitting the challenge into several deployables.
- **Boundary:** CPU-heavy inference still needs **bounded concurrency** and **capacity testing**.

### Docker and Compose
- **Reason:** the challenge asks for a microservice; containers make the runtime **reproducible** and package the API with **PostgreSQL** for reviewers.
- **Boundary:** Docker alone does **not** prove **scalability or production readiness**.

### Readable local fallback
- **Reason:** the demo stays **useful** when hosted-model quotas fail.
- **Boundary:** the UI must **label fallback honestly** and never claim an LLM wrote the answer.