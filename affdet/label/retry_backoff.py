"""Exponential backoff for VLM calls that hit Azure's rate limiter.

Pass 1 of the 1 000-scene list-all labelling run lost 71 scenes to HTTP 429s.
The limiter was the proximate cause, but the retry policy is what turned a
transient refusal into a lost scene: four attempts fired back to back with no
delay, so all four were spent within a couple of seconds while the limiter was
still refusing. With 60 workers tripping the limit on the same burst, they also
retried in lockstep, which keeps the limiter tripped.

Backing off exponentially with jitter fixes both: the wait grows so the limiter
has time to reset, and the jitter spreads the herd so workers do not all return
at the same instant.
"""
from __future__ import annotations

import random

#: Substrings that identify a rate-limit refusal. Matched case-insensitively
#: against the exception text, since the OpenAI SDK surfaces the provider's
#: JSON body rather than a typed error we can branch on reliably.
_RATE_LIMIT_MARKERS = ("429", "rate_limit", "rate limit", "too_many_requests",
                       "too many requests")


def is_rate_limit(exc: BaseException) -> bool:
    """Is this exception a rate-limit refusal rather than a real failure?

    Worth distinguishing: backing off 8 seconds for a malformed-JSON response
    wastes time on a fault that will recur identically, while retrying a 429
    immediately wastes the attempt.
    """
    text = str(exc).lower()
    return any(m in text for m in _RATE_LIMIT_MARKERS)


def backoff_delays(attempts: int, base: float = 1.0, cap: float = 30.0,
                   jitter: float = 0.5, seed: int | None = None
                   ) -> list[float]:
    """Delay before each retry: base * 2**i, capped, with multiplicative jitter.

    `jitter=0.5` spreads each delay over [0.5x, 1.5x] of nominal. Returns one
    delay per attempt so the caller can zip it against its retry loop.
    """
    if attempts < 0:
        raise ValueError(f"attempts must be non-negative, got {attempts}")
    rng = random.Random(seed)
    out = []
    for i in range(attempts):
        d = min(base * (2 ** i), cap)
        if jitter:
            d *= 1.0 + rng.uniform(-jitter, jitter)
        out.append(d)
    return out
