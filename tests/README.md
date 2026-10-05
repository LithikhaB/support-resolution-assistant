# Compact test suite

The regular Python suite contains **50 collected cases**, including **5 PostgreSQL integration cases**. Parameterization means these come from 46 test functions. The five JavaScript console tests run separately.

The previous suite collected 500 cases from 269 functions. This is a deliberately smaller regression suite, not equivalent coverage under a different count. Rare wording variants, repeated malformed-input combinations, low-level configuration boundaries and specialist tooling branches were removed. Git history retains the previous tests. Frozen evaluation reports and held-out data are unchanged.

Core coverage retained:

- API health, readiness failure and conversation request/replay validation.
- Classifier training and abstention; severity versus sentiment; evening/storm evidence and optical-alarm handling.
- Exact source quotes, incomplete procedures, fabricated repairs/citations, conditional remedies and shared outages.
- Follow-up observations, prior actions, recovery and isolation of multiple issues.
- Provider failover, caching/privacy, sanitized errors, rate-limit cooldown and faithfulness rejection.
- Embedding batches, chunk overlap, SQL parameterization, vector evidence, rank fusion and reranking.
- Corpus staging, train-only examples, held-out exclusion, frozen-report protection and honest unrated human reviews.
- Database indexing/updating, rollback on failed writes, repeatable migrations, retrieval and index locking.

Run from the repository root (using its virtual environment):

```powershell
$env:RUN_DB_TESTS = "1"
$env:POSTGRES_HOST = "127.0.0.1"
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.work/pytest-compact -o cache_dir=.work/cache-compact
.\.venv\Scripts\python.exe -m ruff check
.\.venv\Scripts\python.exe -m ruff format --check app scripts tests
node --test tests/web/agent_console.test.cjs
```

Without `RUN_DB_TESTS=1`, the expected result is 45 passing Python cases and 5 skipped database cases. Use a fresh `--basetemp` directory if Windows permissions prevent pytest from reusing a prior directory. Live provider calls, independently rated plan quality, load behavior and production correctness are not established by these tests.
