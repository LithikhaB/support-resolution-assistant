# AI-Powered Support Resolution Assistant

A telecom support assistant designed to turn a raw customer complaint into relevant historical evidence and, eventually, a grounded resolution with citations. It addresses the vocabulary mismatch that makes keyword-only searches miss similar issues.


## What works today

- Download and normalize English support records with source provenance and checksums.
- Preserve different replies to the same complaint while removing exact duplicate records.
- Distinguish historical replies from verified resolutions through validated evidence contracts.
- Prepare a reproducible JSONL corpus and data-quality manifest.
- Run FastAPI liveness and PostgreSQL schema-readiness checks.
- Apply an additive database migration without deleting existing data.

## Architecture


## Data and evidence quality

Source: [Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets), a public, mixed-domain support dataset. Its card lists **CC BY-NC 4.0**. It is not a reviewed telecom knowledge base, and an agent reply does not establish a successful outcome.

| Current local snapshot | Count |
|---|---:|
| English source rows | 28,261 |
| Historical response records | 23,790 |
| Exact duplicate records removed | 4,458 |
| Short bodies / empty replies removed | 7 / 6 |
| Distinct normalized complaint groups | 23,643 |
| Verified resolutions / KB articles | 0 / 0 |





| Current endpoint | Purpose |
|---|---|
| `GET /api/v1/health` | Process liveness |
| `GET /api/v1/ready` | Database/schema readiness; returns 503 if unavailable or incomplete |
| `GET /docs` | Interactive API documentation |

