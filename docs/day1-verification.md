# Day 1 corrections and verification

The ingestion foundation now preserves historical agent replies without asserting that their cases were resolved. Query understanding, retrieval, generation and KB ingestion remain future work.

## Implemented corrections

- Separate `response` from verified `resolution`, and require outcome evidence when claiming a verified resolution.
- Separate source `ticket_type` and `priority` from unknown `intent` and `severity`.
- Preserve anonymization placeholders and normalize literal escaped whitespace.
- Keep different replies and metadata variants; deduplicate only normalized complete records and retain source occurrences.
- Use content-derived SHA-256 document IDs and exact complaint group IDs. Content changes create new identities because no immutable upstream ticket key is available.
- Resolve future dataset downloads to a commit SHA, and record source/output hashes and processing version. The existing CSV's upstream revision remains unknown; it was not replaced or misattributed to today's revision.
- Validate CSV headers, refuse empty output, check raw manifest hashes, and publish each output file atomically.
- Add `.env.example`, database timeouts, separate liveness/readiness, a working database check, and an additive database migration.
- Correct the README's Hit Rate versus Recall definitions and implemented-versus-planned descriptions.

## Local corpus result

| Measure | Result |
|---|---:|
| Raw rows | 28,261 |
| Historical response records | 23,790 |
| Exact duplicates removed | 4,458 |
| Short bodies removed | 7 |
| Empty responses removed | 6 |
| Complaint groups | 23,643 |
| Verified resolutions | 0 |
| Literal newline artifacts remaining | 0 |

The increase from 23,643 to 23,790 documents retains 147 additional reply or metadata variants. It does not add new complaints or invented outcomes.

See `data/processed/manifest.json` for checksums and distributions. That manifest and the dataset are ignored by Git; reproduce them with `python -m scripts.prepare_data` using the recorded source snapshot.

## Verification

**Result: 26 tests passed** on Python 3.14. Existing FastAPI/Starlette dependencies emitted deprecation warnings. All 23,790 processed records passed schema validation, and a repeated full preparation run produced the same output checksum. A fresh SQL schema was also tested in a rolled-back transaction: valid historical replies were accepted and unverified resolution claims were rejected.

Tests cover cleaning, evidence validation, source-row provenance, variant preservation, row-order-independent IDs, rejected CSVs, drop accounting, atomic-write failure, deterministic preparation, checksum mismatch rejection, pinned downloads, and API liveness/readiness failures. External downloads are mocked in tests.

The existing local database was reachable even though Docker was not on PATH. It had zero documents and zero chunks. The additive migration was applied successfully, and the live database readiness check passed. No corpus insertion or retrieval index was created; those belong to Day 2.

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m scripts.prepare_data
.\.venv\Scripts\python.exe -m scripts.check_db
```

## Remaining boundaries

Historical replies are unverified evidence. A schema check cannot prove that a supplied closure reference is true. The corpus still needs reviewed telecom knowledge and successful resolutions. Exact complaint group IDs do not detect near-duplicates. JSONL and its manifest are individually atomic; consumers must verify the output hash to detect an interrupted publication between the two writes. Future ingestion needs a deliberate migration/rebuild from old row-number IDs, not blind mixing with content IDs.

The old synthetic generator is an independent demo fixture generator and is not connected to the Hugging Face ingestion pipeline or used as evaluation ground truth.

## Snapshot checksums

- Source CSV SHA-256: `d0676760a8a512046656fb9a539d9dbfc8513e8c81a7a5e864e95f6bb37fb949`
- Processed JSONL SHA-256: `a1c35aa420437b4fbfa64e817c0cc2d2084ed25c457c654b6f45894c68e1acfd`
- Pipeline version: `2.0.0`
