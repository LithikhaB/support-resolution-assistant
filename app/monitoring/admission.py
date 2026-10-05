"""Bounded FIFO admission per event loop; waiting requests consume no DB connection."""

import asyncio
from collections import deque
from threading import Lock
from weakref import WeakKeyDictionary


class Admission:
    def __init__(self):
        self.active = 0
        self.waiters = deque()

    async def acquire(self, limit, queue_size, timeout):
        if self.active < limit and not self.waiters:
            self.active += 1
            return True
        if len(self.waiters) >= queue_size:
            return False
        future = asyncio.get_running_loop().create_future()
        self.waiters.append(future)
        try:
            await asyncio.wait_for(asyncio.shield(future), timeout)
            return True
        except (TimeoutError, asyncio.CancelledError) as exc:
            if future in self.waiters:
                self.waiters.remove(future)
                future.cancel()
            elif future.done() and not future.cancelled():
                # A slot granted at the timeout/cancellation boundary must be returned.
                self.release()
            if isinstance(exc, asyncio.CancelledError):
                raise
            return False

    def release(self):
        while self.waiters:
            future = self.waiters.popleft()
            if not future.done():
                future.set_result(True)
                return
        self.active -= 1


_loops = WeakKeyDictionary()
_lock = Lock()


def current_admission():
    # Separate test event loops; each deployed worker has one serving event loop.
    loop = asyncio.get_running_loop()
    with _lock:
        if loop not in _loops:
            _loops[loop] = Admission()
        return _loops[loop]
