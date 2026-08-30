"""Retry backoff for the list-all driver.

Pass 1 of the 1 000-scene labelling run lost 71 scenes to HTTP 429s. The cause
was not the rate limit itself but the retry policy: four attempts fired back to
back with no delay, so all four burned out inside a couple of seconds while the
limiter was still refusing. The scenes were recoverable (the driver is
resumable) but the run should not need a second pass to collect them.
"""
import pytest

from affdet.label.retry_backoff import backoff_delays, is_rate_limit


def test_delays_grow_exponentially():
    d = backoff_delays(attempts=4, base=1.0, jitter=0.0)
    assert d == [1.0, 2.0, 4.0, 8.0]


def test_delays_are_capped():
    d = backoff_delays(attempts=6, base=1.0, cap=5.0, jitter=0.0)
    assert max(d) <= 5.0
    assert d == [1.0, 2.0, 4.0, 5.0, 5.0, 5.0]


def test_jitter_stays_within_bounds():
    """Jitter spreads a thundering herd of 60 workers that all tripped the
    limiter on the same request, so they do not retry in lockstep."""
    for _ in range(50):
        d = backoff_delays(attempts=3, base=1.0, jitter=0.5, seed=None)
        for i, v in enumerate(d):
            nominal = 1.0 * (2 ** i)
            assert nominal * 0.5 <= v <= nominal * 1.5


def test_jitter_is_reproducible_under_seed():
    a = backoff_delays(attempts=4, base=1.0, jitter=0.5, seed=7)
    b = backoff_delays(attempts=4, base=1.0, jitter=0.5, seed=7)
    assert a == b


def test_zero_attempts_gives_no_delays():
    assert backoff_delays(attempts=0) == []


def test_negative_attempts_raises():
    with pytest.raises(ValueError):
        backoff_delays(attempts=-1)


def test_rate_limit_detected_from_429_text():
    assert is_rate_limit(Exception("Error code: 429 - rate_limit_exceeded"))
    assert is_rate_limit(Exception("too_many_requests"))


def test_non_rate_limit_errors_are_not_misread():
    """A malformed-JSON failure must not be treated as a rate limit: backing
    off 8 seconds for a bug that will recur identically just wastes time."""
    assert not is_rate_limit(ValueError("response was not JSON"))
    assert not is_rate_limit(Exception("empty response"))
