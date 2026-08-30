import threading
import time

from affdet.label.rate_limiter import TokenRateLimiter


def test_acquire_does_not_block_when_budget_available():
    limiter = TokenRateLimiter(tokens_per_minute=1000, clock=lambda: 0.0)
    start = time.monotonic()
    limiter.acquire(500)
    elapsed = time.monotonic() - start
    assert elapsed < 0.05  # no real sleep needed, budget was available


def test_acquire_blocks_until_bucket_refills():
    now = [0.0]
    sleeps = []

    def fake_sleep(s):
        sleeps.append(s)
        now[0] += s

    limiter = TokenRateLimiter(
        tokens_per_minute=600,  # 10 tokens/sec
        clock=lambda: now[0],
        sleep_fn=fake_sleep,
    )
    limiter.acquire(600)  # drains the whole starting bucket
    limiter.acquire(60)   # needs 6s worth of refill at 10/sec
    assert sleeps  # had to wait
    assert sum(sleeps) >= 5.9


def test_report_usage_reduces_available_budget():
    now = [0.0]
    limiter = TokenRateLimiter(tokens_per_minute=1000, clock=lambda: now[0])
    limiter.acquire(100)  # reserve based on an estimate
    limiter.report_actual(400)  # actual usage was higher than estimate
    assert limiter.available() <= 600 + 1e-6


def test_current_rate_reflects_recent_usage():
    now = [0.0]
    limiter = TokenRateLimiter(tokens_per_minute=1_000_000, clock=lambda: now[0])
    limiter.report_actual(60_000)
    now[0] += 1.0  # one second later
    rate = limiter.current_tokens_per_minute()
    assert rate > 0  # some positive observed rate, not just the configured cap


def test_thread_safety_many_concurrent_acquires():
    limiter = TokenRateLimiter(tokens_per_minute=10_000_000)
    errors = []

    def worker():
        try:
            for _ in range(50):
                limiter.acquire(100)
                limiter.report_actual(100)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
