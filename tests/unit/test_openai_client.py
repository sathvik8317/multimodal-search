"""Tests for the real OpenAI client wrapper.

The underlying SDK object is always injected as a fake. These tests never
construct a real openai.OpenAI and never touch the network.
"""

import httpx
import openai
import pytest
from openai.types import CreateEmbeddingResponse, Embedding
from openai.types.create_embedding_response import Usage

from mmsearch import config
from mmsearch.clients.openai import OpenAIClient
from mmsearch.clients.protocols import EmbedInput, EmbeddingClient

_REQUEST = httpx.Request("POST", "https://api.openai.com/v1/embeddings")


def _embed_response(vectors: list[list[float]]) -> CreateEmbeddingResponse:
    return CreateEmbeddingResponse(
        data=[
            Embedding(embedding=v, index=i, object="embedding") for i, v in enumerate(vectors)
        ],
        model=config.OPENAI_EMBED_MODEL,
        object="list",
        usage=Usage(prompt_tokens=1, total_tokens=1),
    )


def _rate_limit_error() -> openai.RateLimitError:
    return openai.RateLimitError(
        "rate limited", response=httpx.Response(429, request=_REQUEST), body=None
    )


def _server_error() -> openai.InternalServerError:
    return openai.InternalServerError(
        "server error", response=httpx.Response(500, request=_REQUEST), body=None
    )


def _bad_request_error() -> openai.BadRequestError:
    return openai.BadRequestError(
        "bad request", response=httpx.Response(400, request=_REQUEST), body=None
    )


class FakeSDK:
    """Records calls; returns canned responses; can be told to raise N times."""

    def __init__(self):
        self.embed_calls: list[dict] = []
        self._embed_queue: list = []
        self.embeddings = self

    def queue_embed(self, response_or_exc):
        self._embed_queue.append(response_or_exc)

    def create(self, **kwargs):
        self.embed_calls.append(kwargs)
        item = self._embed_queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def sleeps():
    calls: list[float] = []

    def fake_sleep(seconds: float) -> None:
        calls.append(seconds)

    return calls, fake_sleep


# --- protocol conformance -----------------------------------------------------------

def test_openai_client_conforms_to_embedding_client():
    sdk = FakeSDK()
    client = OpenAIClient(sdk=sdk)
    assert isinstance(client, EmbeddingClient)


def test_construction_without_sdk_or_api_key_raises_clear_error(monkeypatch, tmp_path):
    # chdir away from the repo root, same rationale as the Cohere client test:
    # Settings() reads .env directly, independent of os.environ.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIClient()


# --- embed_documents: mapping + batching --------------------------------------------

def test_embed_documents_sends_text_input():
    sdk = FakeSDK()
    sdk.queue_embed(_embed_response([[0.1] * config.OPENAI_EMBED_DIM]))
    client = OpenAIClient(sdk=sdk)

    client.embed_documents([EmbedInput(text="def f(): pass")])

    assert sdk.embed_calls[0]["input"] == ["def f(): pass"]
    assert sdk.embed_calls[0]["model"] == config.OPENAI_EMBED_MODEL


def test_embed_documents_rejects_image_input():
    client = OpenAIClient(sdk=FakeSDK())

    with pytest.raises(ValueError, match="only embeds text"):
        client.embed_documents([EmbedInput(image_bytes=b"\x89PNG raw bytes")])


def test_embed_documents_batches_across_multiple_calls():
    sdk = FakeSDK()
    sdk.queue_embed(
        _embed_response([[0.1] * config.OPENAI_EMBED_DIM, [0.2] * config.OPENAI_EMBED_DIM])
    )
    sdk.queue_embed(_embed_response([[0.3] * config.OPENAI_EMBED_DIM]))
    client = OpenAIClient(sdk=sdk, embed_batch_size=2)

    items = [EmbedInput(text=t) for t in ("a", "b", "c")]
    vectors = client.embed_documents(items)

    assert len(sdk.embed_calls) == 2
    assert sdk.embed_calls[0]["input"] == ["a", "b"]
    assert sdk.embed_calls[1]["input"] == ["c"]
    assert vectors == [
        [0.1] * config.OPENAI_EMBED_DIM,
        [0.2] * config.OPENAI_EMBED_DIM,
        [0.3] * config.OPENAI_EMBED_DIM,
    ]


def test_embed_documents_empty_list_makes_no_calls():
    sdk = FakeSDK()
    client = OpenAIClient(sdk=sdk)
    assert client.embed_documents([]) == []
    assert sdk.embed_calls == []


# --- embed_query ----------------------------------------------------------------------

def test_embed_query_returns_single_vector():
    sdk = FakeSDK()
    sdk.queue_embed(_embed_response([[0.5] * config.OPENAI_EMBED_DIM]))
    client = OpenAIClient(sdk=sdk)

    vector = client.embed_query("what's the retry backoff")

    assert sdk.embed_calls[0]["input"] == ["what's the retry backoff"]
    assert vector == [0.5] * config.OPENAI_EMBED_DIM


# --- retry behavior --------------------------------------------------------------------

def test_transient_error_is_retried_then_succeeds(sleeps):
    calls, fake_sleep = sleeps
    sdk = FakeSDK()
    sdk.queue_embed(_rate_limit_error())
    sdk.queue_embed(_server_error())
    sdk.queue_embed(_embed_response([[0.1] * config.OPENAI_EMBED_DIM]))
    client = OpenAIClient(sdk=sdk, max_retries=3, sleep=fake_sleep)

    vector = client.embed_query("hello")

    assert vector == [0.1] * config.OPENAI_EMBED_DIM
    assert len(sdk.embed_calls) == 3
    assert len(calls) == 2  # slept before the 2nd and 3rd attempts


def test_retry_backoff_is_exponential(sleeps):
    calls, fake_sleep = sleeps
    sdk = FakeSDK()
    sdk.queue_embed(_rate_limit_error())
    sdk.queue_embed(_rate_limit_error())
    sdk.queue_embed(_embed_response([[0.1] * config.OPENAI_EMBED_DIM]))
    client = OpenAIClient(sdk=sdk, max_retries=3, backoff_seconds=1.0, sleep=fake_sleep)

    client.embed_query("hello")

    assert calls == [1.0, 2.0]


def test_exhausting_retries_reraises_the_transient_error(sleeps):
    _, fake_sleep = sleeps
    sdk = FakeSDK()
    for _ in range(4):
        sdk.queue_embed(_rate_limit_error())
    client = OpenAIClient(sdk=sdk, max_retries=3, sleep=fake_sleep)

    with pytest.raises(openai.RateLimitError):
        client.embed_query("hello")

    assert len(sdk.embed_calls) == 4  # 1 initial + 3 retries


def test_non_transient_error_is_not_retried(sleeps):
    calls, fake_sleep = sleeps
    sdk = FakeSDK()
    sdk.queue_embed(_bad_request_error())
    client = OpenAIClient(sdk=sdk, max_retries=3, sleep=fake_sleep)

    with pytest.raises(openai.BadRequestError):
        client.embed_query("hello")

    assert len(sdk.embed_calls) == 1
    assert calls == []
