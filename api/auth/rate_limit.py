import time
from collections import defaultdict


class RateLimiter:
    def __init__(self, limit: int = 10, window: int = 60) -> None:
        self._limit = limit
        self._window = window
        self._hits: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, ip: str) -> bool:
        now = time.monotonic()
        self._hits[ip] = [t for t in self._hits[ip] if now - t < self._window]
        self._hits[ip].append(now)
        return len(self._hits[ip]) <= self._limit
