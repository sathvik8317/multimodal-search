from fastapi.testclient import TestClient

from mmsearch.api.main import create_app
from mmsearch.retrieve.types import SearchResult
from mmsearch.schema import Modality, TextSource

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


# --- /search ---------------------------------------------------------------------------

def test_search_returns_expected_json_shape_and_length(tmp_path):
    app = create_app(fake_search_fn, thumbnails_dir=tmp_path)
    client = TestClient(app)

    response = client.get("/search", params={"q": "auth token flow", "k": 1})

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

    app = create_app(fake_search_fn, thumbnails_dir=tmp_path)
    client = TestClient(app)

    response = client.get("/search", params={"q": "auth token flow"})

    assert response.status_code == 200
    # fake_search_fn returns min(k, len(CANNED_RESULTS)); with default TOP_K=5
    # and only 2 canned results, expect both back.
    assert len(response.json()) == min(config.TOP_K, len(CANNED_RESULTS))


# --- /healthz ----------------------------------------------------------------------------

def test_healthz_returns_ok(tmp_path):
    app = create_app(fake_search_fn, thumbnails_dir=tmp_path)
    client = TestClient(app)

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --- /thumbnails static mount --------------------------------------------------------------

def test_thumbnails_serves_real_file(tmp_path):
    thumb_file = tmp_path / "auth.png"
    thumb_file.write_bytes(b"fake-png-bytes")

    app = create_app(fake_search_fn, thumbnails_dir=tmp_path)
    client = TestClient(app)

    response = client.get("/thumbnails/auth.png")

    assert response.status_code == 200
    assert response.content == b"fake-png-bytes"


# --- /ui static mount -----------------------------------------------------------------------

def test_ui_returns_html_with_search_input_and_fetch_call(tmp_path):
    app = create_app(fake_search_fn, thumbnails_dir=tmp_path)
    client = TestClient(app)

    response = client.get("/ui")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    body = response.text
    assert "<input" in body
    assert "/search" in body
    assert "fetch(" in body
