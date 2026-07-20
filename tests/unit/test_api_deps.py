import pytest
from fastapi import HTTPException

from mmsearch.api import deps
from mmsearch.settings import Settings


def _settings(**overrides) -> Settings:
    defaults = dict(_env_file=None, api_key="correct-key")
    defaults.update(overrides)
    return Settings(**defaults)


# --- require_api_key --------------------------------------------------------------------

def test_require_api_key_accepts_correct_header():
    deps.require_api_key(x_api_key="correct-key", mm_api_key=None, settings=_settings())


def test_require_api_key_accepts_correct_cookie():
    deps.require_api_key(x_api_key=None, mm_api_key="correct-key", settings=_settings())


def test_require_api_key_prefers_header_when_both_present():
    deps.require_api_key(
        x_api_key="correct-key", mm_api_key="wrong-key", settings=_settings()
    )


def test_require_api_key_rejects_missing_key():
    with pytest.raises(HTTPException) as exc_info:
        deps.require_api_key(x_api_key=None, mm_api_key=None, settings=_settings())
    assert exc_info.value.status_code == 401


def test_require_api_key_rejects_wrong_header():
    with pytest.raises(HTTPException) as exc_info:
        deps.require_api_key(x_api_key="wrong-key", mm_api_key=None, settings=_settings())
    assert exc_info.value.status_code == 401


def test_require_api_key_rejects_wrong_cookie():
    with pytest.raises(HTTPException) as exc_info:
        deps.require_api_key(x_api_key=None, mm_api_key="wrong-key", settings=_settings())
    assert exc_info.value.status_code == 401


def test_require_api_key_rejects_empty_supplied_key_when_configured_key_unset():
    # env="staging" doesn't require api_key to be set (unlike "dev"); a misconfigured
    # deployment with no gate secret must still deny, never treat "" == "" as a match.
    unset = _settings(env="staging", api_key=None)

    with pytest.raises(HTTPException) as exc_info:
        deps.require_api_key(x_api_key="", mm_api_key=None, settings=unset)
    assert exc_info.value.status_code == 401


# --- rate_limit --------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_rate_limit_state():
    deps._hits.clear()
    yield
    deps._hits.clear()


def test_rate_limit_allows_calls_under_the_max():
    settings = _settings(rate_limit_max=3, rate_limit_window=60.0)

    deps.rate_limit(settings=settings)
    deps.rate_limit(settings=settings)
    deps.rate_limit(settings=settings)


def test_rate_limit_blocks_the_call_over_the_max():
    settings = _settings(rate_limit_max=3, rate_limit_window=60.0)

    for _ in range(3):
        deps.rate_limit(settings=settings)

    with pytest.raises(HTTPException) as exc_info:
        deps.rate_limit(settings=settings)
    assert exc_info.value.status_code == 429


def test_rate_limit_allows_again_after_window_elapses(monkeypatch):
    settings = _settings(rate_limit_max=2, rate_limit_window=60.0)
    now = [1000.0]
    monkeypatch.setattr(deps.time, "monotonic", lambda: now[0])

    deps.rate_limit(settings=settings)
    deps.rate_limit(settings=settings)
    with pytest.raises(HTTPException):
        deps.rate_limit(settings=settings)

    now[0] += 61.0  # past the window
    deps.rate_limit(settings=settings)  # should not raise


def test_rate_limit_does_not_grow_unbounded_across_window_boundary(monkeypatch):
    settings = _settings(rate_limit_max=2, rate_limit_window=60.0)
    now = [1000.0]
    monkeypatch.setattr(deps.time, "monotonic", lambda: now[0])

    deps.rate_limit(settings=settings)
    now[0] += 61.0
    deps.rate_limit(settings=settings)

    assert len(deps._hits) == 1
