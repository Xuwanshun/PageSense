from api.auth.rate_limit import RateLimiter


def test_allows_requests_under_limit():
    rl = RateLimiter(limit=3, window=60)
    assert rl.is_allowed("1.2.3.4") is True
    assert rl.is_allowed("1.2.3.4") is True
    assert rl.is_allowed("1.2.3.4") is True


def test_blocks_on_limit_exceeded():
    rl = RateLimiter(limit=3, window=60)
    for _ in range(3):
        rl.is_allowed("1.2.3.4")
    assert rl.is_allowed("1.2.3.4") is False


def test_different_ips_are_independent():
    rl = RateLimiter(limit=1, window=60)
    rl.is_allowed("10.0.0.1")
    assert rl.is_allowed("10.0.0.1") is False
    assert rl.is_allowed("10.0.0.2") is True
