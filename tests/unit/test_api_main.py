import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from mmsearch.api import deps
from mmsearch.api.main import _resolve_thumbnail, create_app
from mmsearch.retrieve.types import SearchResult
from mmsearch.schema import Modality, TextSource
from mmsearch.settings import Settings, get_settings

TEST_API_KEY = "test-secret-key"

CANNED_RESULTS = [
    SearchResult(
        id="img:auth.png",
        modality=Modality.DIAGRAM,
        score=0.9,
        snippet="diagram showing the auth token flow",
        thumbnail_ref="auth.png",
        source_path="docs/auth.png",
        text_source=TextSource.VLM_CAPTION,
    ),
    SearchResult(
        id="code:a.py#f",
        modality=Modality.CODE,
        score=0.5,
        snippet="the retry backoff is exponential",
        thumbnail_ref="",
        source_path="src/a.py",
        text_source=TextSource.CODE_SOURCE,
    ),
]


def fake_search_fn(query: str, k: int = 5) -> list[SearchResult]:
    return CANNED_RESULTS[:k]


def _test_settings(**overrides) -> Settings:
    defaults = dict(
        _env_file=None, api_key=TEST_API_KEY, rate_limit_max=1000, rate_limit_window=60.0
    )
    defaults.update(overrides)
    return Settings(**defaults)


# --- /search ---------------------------------------------------------------------------

def test_search_returns_expected_json_shape_and_length(tmp_path):
    app = create_app(fake_search_fn, thumbnails_dir=tmp_path, settings=_test_settings())
    client = TestClient(app)

    response = client.get(
        "/search",
        params={"q": "auth token flow", "k": 1},
        headers={"X-API-Key": TEST_API_KEY},
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    result = body[0]
    assert result["id"] == "img:auth.png"
    assert result["modality"] == "diagram"
    assert result["score"] == 0.9
    assert result["snippet"] == "diagram showing the auth token flow"
    assert result["thumbnail_ref"] == "auth.png"
    assert result["source_path"] == "docs/auth.png"
    assert result["text_source"] == "vlm_caption"


def test_search_defaults_k_to_top_k(tmp_path):
    from mmsearch import config

    app = create_app(fake_search_fn, thumbnails_dir=tmp_path, settings=_test_settings())
    client = TestClient(app)

    response = client.get(
        "/search", params={"q": "auth token flow"}, headers={"X-API-Key": TEST_API_KEY}
    )

    assert response.status_code == 200
    # fake_search_fn returns min(k, len(CANNED_RESULTS)); with default TOP_K=5
    # and only 2 canned results, expect both back.
    assert len(response.json()) == min(config.TOP_K, len(CANNED_RESULTS))


# --- /healthz ----------------------------------------------------------------------------

def test_healthz_returns_ok(tmp_path):
    app = create_app(fake_search_fn, thumbnails_dir=tmp_path, settings=_test_settings())
    client = TestClient(app)

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --- /thumbnails static mount --------------------------------------------------------------

def test_thumbnails_serves_real_file(tmp_path):
    thumb_file = tmp_path / "auth.png"
    thumb_file.write_bytes(b"fake-png-bytes")

    app = create_app(fake_search_fn, thumbnails_dir=tmp_path, settings=_test_settings())
    client = TestClient(app)

    response = client.get("/thumbnails/auth.png", headers={"X-API-Key": TEST_API_KEY})

    assert response.status_code == 200
    assert response.content == b"fake-png-bytes"


# --- /ui static mount -----------------------------------------------------------------------
#
# Deliberately untested. /ui serves the Vite build output (see FRONTEND_PLAN.md),
# which is gitignored and absent until `npm run build` runs -- so any assertion
# about its contents fails on a fresh clone and in CI. The previous test here
# asserted the body contained "<input" and "fetch(", which stopped being true the
# moment the UI became a bundled artifact rather than a hand-written file. What
# remains is one line of framework mount config with nothing project-specific to
# assert. The /thumbnails mount above is still tested: it serves real app data.


# --- API-key gate -------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    deps._hits.clear()
    yield
    deps._hits.clear()


def test_search_without_key_returns_401_and_never_calls_search_fn(tmp_path):
    calls: list[str] = []

    def spy_search_fn(query: str, k: int = 5) -> list[SearchResult]:
        calls.append(query)
        return CANNED_RESULTS[:k]

    app = create_app(spy_search_fn, thumbnails_dir=tmp_path, settings=_test_settings())
    client = TestClient(app)

    response = client.get("/search", params={"q": "auth token flow"})

    assert response.status_code == 401
    assert calls == []  # the expensive Cohere-backed pipeline never ran


def test_search_with_wrong_key_returns_401(tmp_path):
    app = create_app(fake_search_fn, thumbnails_dir=tmp_path, settings=_test_settings())
    client = TestClient(app)

    response = client.get(
        "/search", params={"q": "auth token flow"}, headers={"X-API-Key": "wrong-key"}
    )

    assert response.status_code == 401


def test_search_with_correct_key_header_returns_200(tmp_path):
    app = create_app(fake_search_fn, thumbnails_dir=tmp_path, settings=_test_settings())
    client = TestClient(app)

    response = client.get(
        "/search", params={"q": "auth token flow"}, headers={"X-API-Key": TEST_API_KEY}
    )

    assert response.status_code == 200


def test_thumbnails_without_key_returns_401(tmp_path):
    (tmp_path / "auth.png").write_bytes(b"fake-png-bytes")
    app = create_app(fake_search_fn, thumbnails_dir=tmp_path, settings=_test_settings())
    client = TestClient(app)

    response = client.get("/thumbnails/auth.png")

    assert response.status_code == 401


def test_thumbnails_with_cookie_only_returns_200(tmp_path):
    # The <img> path: browsers cannot attach a custom X-API-Key header, only cookies.
    (tmp_path / "auth.png").write_bytes(b"fake-png-bytes")
    app = create_app(fake_search_fn, thumbnails_dir=tmp_path, settings=_test_settings())
    client = TestClient(app)
    client.cookies.set("mm_api_key", TEST_API_KEY)

    response = client.get("/thumbnails/auth.png")

    assert response.status_code == 200
    assert response.content == b"fake-png-bytes"


def test_thumbnails_nonexistent_file_returns_404(tmp_path):
    app = create_app(fake_search_fn, thumbnails_dir=tmp_path, settings=_test_settings())
    client = TestClient(app)

    response = client.get("/thumbnails/nope.png", headers={"X-API-Key": TEST_API_KEY})

    assert response.status_code == 404


def test_thumbnails_serves_uploaded_file_from_r2_storage(tmp_path):
    class _FakeR2Storage:
        def get_bytes(self, key: str) -> bytes:
            assert key == "uploads/alice/img.png"
            return b"r2-png-bytes"

    app = create_app(
        fake_search_fn,
        thumbnails_dir=tmp_path,
        settings=_test_settings(),
        upload_thumbnail_storage=_FakeR2Storage(),
    )
    client = TestClient(app)

    response = client.get(
        "/thumbnails/uploads/alice/img.png", headers={"X-API-Key": TEST_API_KEY}
    )

    assert response.status_code == 200
    assert response.content == b"r2-png-bytes"
    assert response.headers["content-type"] == "image/png"


def test_thumbnails_uploads_prefix_missing_key_returns_404(tmp_path):
    class _EmptyR2Storage:
        def get_bytes(self, key: str) -> bytes:
            raise FileNotFoundError(key)

    app = create_app(
        fake_search_fn,
        thumbnails_dir=tmp_path,
        settings=_test_settings(),
        upload_thumbnail_storage=_EmptyR2Storage(),
    )
    client = TestClient(app)

    response = client.get(
        "/thumbnails/uploads/alice/missing.png", headers={"X-API-Key": TEST_API_KEY}
    )

    assert response.status_code == 404


def test_thumbnails_uploads_prefix_without_storage_configured_returns_404(tmp_path):
    app = create_app(fake_search_fn, thumbnails_dir=tmp_path, settings=_test_settings())
    client = TestClient(app)

    response = client.get(
        "/thumbnails/uploads/alice/img.png", headers={"X-API-Key": TEST_API_KEY}
    )

    assert response.status_code == 404


def test_thumbnails_local_path_unaffected_by_upload_storage_present(tmp_path):
    (tmp_path / "auth.png").write_bytes(b"local-png-bytes")

    class _UnusedR2Storage:
        def get_bytes(self, key: str) -> bytes:
            raise AssertionError("should not be called for a non-uploads/ path")

    app = create_app(
        fake_search_fn,
        thumbnails_dir=tmp_path,
        settings=_test_settings(),
        upload_thumbnail_storage=_UnusedR2Storage(),
    )
    client = TestClient(app)

    response = client.get("/thumbnails/auth.png", headers={"X-API-Key": TEST_API_KEY})

    assert response.status_code == 200
    assert response.content == b"local-png-bytes"


def test_healthz_requires_no_key(tmp_path):
    # /healthz itself has no dependency on settings, but create_app() now always
    # touches settings once (for CORS), so this still needs to pass a valid one.
    app = create_app(fake_search_fn, thumbnails_dir=tmp_path, settings=_test_settings())
    client = TestClient(app)

    response = client.get("/healthz")

    assert response.status_code == 200


# --- /thumbnails path-containment (trust boundary) -----------------------------------------
#
# Tested against the pure resolver function, not through TestClient: httpx normalizes
# ".." segments out of request URLs before sending, which would make an HTTP-level
# traversal test pass even with a broken containment check. This is the real check.

def test_resolve_thumbnail_returns_path_for_existing_file(tmp_path):
    thumbnails_root = tmp_path / "thumbnails"
    thumbnails_root.mkdir()
    (thumbnails_root / "auth.png").write_bytes(b"data")

    resolved = _resolve_thumbnail(thumbnails_root, "auth.png")

    assert resolved == (thumbnails_root / "auth.png").resolve()


def test_resolve_thumbnail_supports_nested_paths(tmp_path):
    thumbnails_root = tmp_path / "thumbnails"
    (thumbnails_root / "sub").mkdir(parents=True)
    (thumbnails_root / "sub" / "a.png").write_bytes(b"data")

    resolved = _resolve_thumbnail(thumbnails_root, "sub/a.png")

    assert resolved.is_file()


def test_resolve_thumbnail_rejects_traversal_outside_root(tmp_path):
    thumbnails_root = tmp_path / "thumbnails"
    thumbnails_root.mkdir()
    (tmp_path / "secret.txt").write_text("do not leak")

    with pytest.raises(HTTPException) as exc_info:
        _resolve_thumbnail(thumbnails_root, "../secret.txt")
    assert exc_info.value.status_code == 404


def test_resolve_thumbnail_rejects_backslash_traversal(tmp_path):
    # Not a URL path separator, but pathlib treats it as one once joined -- a
    # real Windows-specific vector, and this project runs on Windows.
    thumbnails_root = tmp_path / "thumbnails"
    thumbnails_root.mkdir()
    (tmp_path / "secret.txt").write_text("do not leak")

    with pytest.raises(HTTPException) as exc_info:
        _resolve_thumbnail(thumbnails_root, "..\\secret.txt")
    assert exc_info.value.status_code == 404


def test_resolve_thumbnail_rejects_absolute_path_escape(tmp_path):
    thumbnails_root = tmp_path / "thumbnails"
    thumbnails_root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("nope")

    # Joining an absolute path onto a Path discards the base entirely -- must
    # still be rejected by the containment check, not silently served.
    with pytest.raises(HTTPException) as exc_info:
        _resolve_thumbnail(thumbnails_root, str(outside))
    assert exc_info.value.status_code == 404


def test_resolve_thumbnail_rejects_missing_file(tmp_path):
    thumbnails_root = tmp_path / "thumbnails"
    thumbnails_root.mkdir()

    with pytest.raises(HTTPException) as exc_info:
        _resolve_thumbnail(thumbnails_root, "nope.png")
    assert exc_info.value.status_code == 404
