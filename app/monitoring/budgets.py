"""Fail-fast token budgets with an optional shared Postgres backend."""

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from math import ceil
from threading import Lock
from time import monotonic

from app.config.settings import get_settings
from app.database.connection import get_connection
from app.llm.contracts import LanguageUnavailable

reserved_calls = ContextVar("reserved_provider_calls", default=None)


class RequestBudgets:
    """Bound memory and DB retention; use DB time and atomic upserts across replicas."""

    def __init__(self, clock=monotonic, connection=get_connection):
        self.clock, self.connection = clock, connection
        self.lock = Lock()
        self.items = OrderedDict()
        self.operations = 0

    def take(self, key, rate, burst, *, backend="memory", cost=1):
        if cost > burst:
            return max(1, ceil(60 * cost / rate))
        if backend == "postgres":
            with self.connection() as conn:
                row = conn.execute(
                    """INSERT INTO support_request_budgets(bucket_key,tokens)
                       VALUES (%s,%s) ON CONFLICT(bucket_key) DO UPDATE SET
                       tokens=least(%s,support_request_budgets.tokens +
                         greatest(0,extract(epoch FROM (clock_timestamp()-support_request_budgets.updated_at))) * %s)-%s,
                       updated_at=clock_timestamp()
                       WHERE support_request_budgets.blocked_until<=clock_timestamp()
                         AND least(%s,support_request_budgets.tokens +
                         greatest(0,extract(epoch FROM (clock_timestamp()-support_request_budgets.updated_at))) * %s)>=%s
                       RETURNING tokens""",
                    (key, burst - cost, burst, rate / 60, cost, burst, rate / 60, cost),
                ).fetchone()
                with self.lock:
                    self.operations += 1
                    cleanup = self.operations % 100 == 0
                if cleanup:
                    conn.execute(
                        "DELETE FROM support_request_budgets WHERE updated_at<clock_timestamp()-interval '1 day' AND blocked_until<=clock_timestamp()"
                    )
                if row:
                    return 0
                blocked = conn.execute(
                    "SELECT CASE WHEN blocked_until<=clock_timestamp() THEN 0 ELSE extract(epoch FROM (blocked_until-clock_timestamp())) END FROM support_request_budgets WHERE bucket_key=%s",
                    (key,),
                ).fetchone()
                return max(1, ceil(60 * cost / rate), ceil(blocked[0]) if blocked else 0)
        now = self.clock()
        with self.lock:
            tokens, last, blocked = self.items.get(key, (float(burst), now, 0))
            tokens = min(burst, tokens + max(0, now - last) * rate / 60)
            retry = max(0, blocked - now, (cost - tokens) * 60 / rate)
            self.items[key] = (tokens - cost if retry <= 0 else tokens, now, blocked)
            self.items.move_to_end(key)
            while len(self.items) > 4096:
                self.items.popitem(last=False)
            return ceil(retry)

    def block(self, key, seconds, *, backend="memory"):
        seconds = min(86400, max(1, seconds))
        if backend == "postgres":
            with self.connection() as conn:
                conn.execute(
                    """INSERT INTO support_request_budgets(bucket_key,tokens,blocked_until)
                       VALUES (%s,0,clock_timestamp()+%s*interval '1 second')
                       ON CONFLICT(bucket_key) DO UPDATE SET
                         blocked_until=greatest(support_request_budgets.blocked_until,excluded.blocked_until),
                         updated_at=clock_timestamp()""",
                    (key, seconds),
                )
            return
        with self.lock:
            now = self.clock()
            tokens, last, blocked = self.items.get(key, (0, now, 0))
            self.items[key] = (tokens, last, max(blocked, now + seconds))
            self.items.move_to_end(key)
            while len(self.items) > 4096:
                self.items.popitem(last=False)


budgets = RequestBudgets()


def provider_budget(provider, cost=1):
    """Namespace provider quotas separately from API admission budgets."""
    settings = provider.settings
    key = "provider:" + provider.name
    reserved = reserved_calls.get()
    if reserved and reserved["key"] == key and reserved["remaining"]:
        reserved["remaining"] -= 1
        return key, 0
    retry = budgets.take(
        key,
        settings.llm_requests_per_minute,
        settings.llm_request_burst,
        backend=settings.rate_limit_backend,
        cost=cost,
    )
    return key, retry


@contextmanager
def reserve_generation(provider, cost):
    """Admit the generation and critique together, avoiding a wasted first call."""
    key, retry = provider_budget(provider, cost=cost)
    if retry:
        raise LanguageUnavailable("provider_throttled", retry_after=retry)
    token = reserved_calls.set({"key": key, "remaining": cost})
    try:
        yield
    finally:
        reserved_calls.reset(token)


def api_budget(client):
    settings = get_settings()
    for key, rate, burst in (
        ("api:global", settings.api_global_requests_per_minute, settings.api_global_request_burst),
        ("api:client:" + client, settings.api_requests_per_minute, settings.api_request_burst),
    ):
        retry = budgets.take(key, rate, burst, backend=settings.rate_limit_backend)
        if retry:
            return retry
    return 0
