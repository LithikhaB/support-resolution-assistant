"""Bounded FIFO waiting, timeout/cancellation cleanup and atomic API budgets."""

import asyncio

from app.monitoring.admission import Admission
from app.monitoring.budgets import RequestBudgets


def test_fifo_waiting_keeps_execution_bounded():
    async def scenario():
        gate = Admission()
        assert await gate.acquire(1, 2, 1)
        first = asyncio.create_task(gate.acquire(1, 2, 1))
        await asyncio.sleep(0)
        second = asyncio.create_task(gate.acquire(1, 2, 1))
        await asyncio.sleep(0)
        assert not await gate.acquire(1, 2, 1)
        assert gate.active == 1 and len(gate.waiters) == 2
        gate.release()
        assert await first
        assert not second.done() and gate.active == 1
        gate.release()
        assert await second
        gate.release()
        assert gate.active == 0 and not gate.waiters

    asyncio.run(scenario())


def test_timeout_and_cancellation_do_not_leak_slots():
    async def scenario():
        gate = Admission()
        assert await gate.acquire(1, 2, 1)
        assert not await gate.acquire(1, 2, 0.01)
        waiting = asyncio.create_task(gate.acquire(1, 2, 1))
        await asyncio.sleep(0)
        waiting.cancel()
        try:
            await waiting
        except asyncio.CancelledError:
            pass
        assert not gate.waiters and gate.active == 1
        # Cancellation after release grants a slot must also return it.
        waiting = asyncio.create_task(gate.acquire(1, 2, 1))
        await asyncio.sleep(0)
        gate.release()
        waiting.cancel()
        try:
            await waiting
        except asyncio.CancelledError:
            pass
        assert gate.active == 0 and not gate.waiters
        assert await gate.acquire(1, 2, 1)
        gate.release()

    asyncio.run(scenario())


def test_denied_client_does_not_spend_global_tokens():
    limiter = RequestBudgets(clock=lambda: 0)
    limits = [("global", 120, 4), ("client-a", 60, 1)]
    assert limiter.take_many(limits) == 0
    assert limiter.take_many(limits) == 1
    assert limiter.items["global"][0] == 3
    assert limiter.take_many([("global", 120, 4), ("client-b", 60, 1)]) == 0
    assert limiter.items["global"][0] == 2
