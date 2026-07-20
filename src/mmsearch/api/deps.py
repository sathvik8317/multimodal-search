"""FastAPI dependencies for the API-key gate and search rate limiting."""

from __future__ import annotations

import hmac
import time
from collections import deque

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
_hits: deque[float] = deque()


def rate_limit(settings: Settings = Depends(get_settings)) -> None:
    now = time.monotonic()
    while _hits and now - _hits[0] > settings.rate_limit_window:
        _hits.popleft()
    if len(_hits) >= settings.rate_limit_max:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="rate limited")
    _hits.append(now)
