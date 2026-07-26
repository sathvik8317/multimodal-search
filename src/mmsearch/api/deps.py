"""FastAPI dependencies for the API-key gate and rate limiting."""

from __future__ import annotations

import hmac
import time
from collections import deque
from collections.abc import Callable

from fastapi import Cookie, Depends, Header, HTTPException, status

from mmsearch.settings import Settings, get_settings


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    mm_api_key: str | None = Cookie(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    supplied = x_api_key or mm_api_key or ""
    # settings.api_key can be unset outside "dev" (see Settings); an unset gate
    # secret must always deny, never compare "" == "" as a match.
    if not settings.api_key or not hmac.compare_digest(supplied, settings.api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or missing API key")


# ponytail: single-process in-memory counter with a single global cap (not
# per-client) -- correct for this deployment (one Cohere key, one user), but
# does not hold across multiple uvicorn workers. Swap for slowapi + Redis if
# this ever runs multi-worker.
def _make_rate_limiter(
    hits: deque[float],
    get_max: Callable[[Settings], int],
    get_window: Callable[[Settings], float],
) -> Callable[[Settings], None]:
    def limiter(settings: Settings = Depends(get_settings)) -> None:
        now = time.monotonic()
        window = get_window(settings)
        while hits and now - hits[0] > window:
            hits.popleft()
        if len(hits) >= get_max(settings):
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="rate limited")
        hits.append(now)

    return limiter


_hits: deque[float] = deque()
rate_limit = _make_rate_limiter(_hits, lambda s: s.rate_limit_max, lambda s: s.rate_limit_window)

# Separate counter/budget from _hits: ingestion (embeddings + captioning +
# storage writes) is far more expensive per request than a search, so
# exhausting one must never affect the other.
_upload_hits: deque[float] = deque()
upload_rate_limit = _make_rate_limiter(
    _upload_hits, lambda s: s.upload_rate_limit_max, lambda s: s.upload_rate_limit_window
)
