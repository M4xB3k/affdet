"""Thread-safe token-bucket limiter for staying under an API TPM quota.

Concurrent workers call acquire(estimated_tokens) before each request (blocks
until budget is available) and report_actual(real_tokens) after, so the
bucket reflects true usage instead of the (often much smaller) estimate --
important for VLM image calls where the estimate is a guess but the real
prompt token count is known only after the response comes back.
"""
from __future__ import annotations

import threading
import time
from collections import deque


class TokenRateLimiter:
    def __init__(
        self,
        tokens_per_minute: float,
        clock=time.monotonic,
        sleep_fn=time.sleep,
        window_seconds: float = 60.0,
    ):
        self._capacity = tokens_per_minute
        self._rate_per_sec = tokens_per_minute / 60.0
        self._clock = clock
        self._sleep = sleep_fn
        self._window = window_seconds

        self._tokens = tokens_per_minute  # start with a full bucket
        self._last_refill = clock()
        self._lock = threading.Lock()

        self._usage_log: deque[tuple[float, float]] = deque()  # (timestamp, amount)

    def _refill_locked(self) -> None:
        now = self._clock()
        elapsed = max(0.0, now - self._last_refill)
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate_per_sec)
        self._last_refill = now

    def acquire(self, estimated_tokens: float) -> None:
        while True:
            with self._lock:
                self._refill_locked()
                if self._tokens >= estimated_tokens:
                    self._tokens -= estimated_tokens
                    return
                deficit = estimated_tokens - self._tokens
                wait = deficit / self._rate_per_sec
            self._sleep(wait)

    def report_actual(self, actual_tokens: float) -> None:
        with self._lock:
            self._refill_locked()
            self._tokens -= actual_tokens
            now = self._clock()
            self._usage_log.append((now, actual_tokens))
            self._prune_locked(now)

    def available(self) -> float:
        with self._lock:
            self._refill_locked()
            return self._tokens

    def _prune_locked(self, now: float) -> None:
        cutoff = now - self._window
        while self._usage_log and self._usage_log[0][0] < cutoff:
            self._usage_log.popleft()

    def current_tokens_per_minute(self) -> float:
        with self._lock:
            now = self._clock()
            self._prune_locked(now)
            if not self._usage_log:
                return 0.0
            span = max(now - self._usage_log[0][0], 1e-6)
            total = sum(amount for _, amount in self._usage_log)
            return total * (60.0 / span) if span < self._window else total
