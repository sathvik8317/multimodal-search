"""Tests for the real Cohere client wrapper.

The underlying SDK object is always injected as a fake. These tests never
construct a real cohere.ClientV2 and never touch the network.
"""

import base64

import cohere
import pytest
from cohere.types import (
    EmbedByTypeResponse,
    EmbedByTypeResponseEmbeddings,
    ImageUrlEmbedContent,
    TextEmbedContent,
)
from cohere.v2.types.v2rerank_response import V2RerankResponse
from cohere.v2.types.v2rerank_response_results_item import V2RerankResponseResultsItem

from mmsearch import config
from mmsearch.clients.cohere import CohereClient
from mmsearch.clients.protocols import EmbeddingClient, EmbedInput, Reranker


def _embed_response(vectors: list[list[float]]) -> EmbedByTypeResponse:
    return EmbedByTypeResponse(id="x", embeddings=EmbedByTypeResponseEmbeddings(float=vectors))


def _rerank_response(pairs: list[tuple[int, float]]) -> V2RerankResponse:
    return V2RerankResponse(
        results=[V2RerankResponseResultsItem(index=i, relevance_score=s) for i, s in pairs]
    )


class FakeSDK:
    """Records calls; returns canned responses; can be told to raise N times."""

    def __init__(self):
        self.embed_calls: list[dict] = []
        self.rerank_calls: list[dict] = []
        self._embed_queue: list = []
        self._rerank_queue: list = []

    def queue_embed(self, response_or_exc):
        self._embed_queue.append(response_or_exc)

    def queue_rerank(self, response_or_exc):
        self._rerank_queue.append(response_or_exc)

    def embed(self, **kwargs):
        self.embed_calls.append(kwargs)
        item = self._embed_queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def rerank(self, **kwargs):
        self.rerank_calls.append(kwargs)
        item = self._rerank_queue.pop(0)
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

def test_cohere_client_conforms_to_embedding_client_and_reranker():
    sdk = FakeSDK()
    client = CohereClient(sdk=sdk)
    assert isinstance(client, EmbeddingClient)
    assert isinstance(client, Reranker)


def test_construction_without_sdk_or_api_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="COHERE_API_KEY"):
        CohereClient()


# --- embed_documents: mapping + batching --------------------------------------------

def test_embed_documents_maps_text_input_to_text_embed_content():
    sdk = FakeSDK()
    sdk.queue_embed(_embed_response([[0.1] * config.EMBED_DIM]))
    client = CohereClient(sdk=sdk)

    client.embed_documents([EmbedInput(text="def f(): pass")])

    sent_inputs = sdk.embed_calls[0]["inputs"]
    assert len(sent_inputs) == 1
    assert isinstance(sent_inputs[0].content[0], TextEmbedContent)
    assert sent_inputs[0].content[0].text == "def f(): pass"


def test_embed_documents_maps_image_input_to_base64_data_uri():
    sdk = FakeSDK()
    sdk.queue_embed(_embed_response([[0.1] * config.EMBED_DIM]))
    client = CohereClient(sdk=sdk)

    image_bytes = b"\x89PNG raw bytes"
    client.embed_documents([EmbedInput(image_bytes=image_bytes)])

    sent_inputs = sdk.embed_calls[0]["inputs"]
    content = sent_inputs[0].content[0]
    assert isinstance(content, ImageUrlEmbedContent)
    expected_b64 = base64.b64encode(image_bytes).decode("ascii")
    assert content.image_url.url == f"data:image/png;base64,{expected_b64}"


def test_embed_documents_uses_search_document_input_type():
    sdk = FakeSDK()
    sdk.queue_embed(_embed_response([[0.1] * config.EMBED_DIM]))
    client = CohereClient(sdk=sdk)

    client.embed_documents([EmbedInput(text="hello")])

    assert sdk.embed_calls[0]["input_type"] == "search_document"


def test_embed_documents_batches_across_multiple_calls():
    sdk = FakeSDK()
    sdk.queue_embed(_embed_response([[0.1] * config.EMBED_DIM, [0.2] * config.EMBED_DIM]))
    sdk.queue_embed(_embed_response([[0.3] * config.EMBED_DIM]))
    client = CohereClient(sdk=sdk, embed_batch_size=2)

    items = [EmbedInput(text=t) for t in ("a", "b", "c")]
    vectors = client.embed_documents(items)

    assert len(sdk.embed_calls) == 2
    assert len(sdk.embed_calls[0]["inputs"]) == 2
    assert len(sdk.embed_calls[1]["inputs"]) == 1
    assert vectors == [[0.1] * config.EMBED_DIM, [0.2] * config.EMBED_DIM, [0.3] * config.EMBED_DIM]


def test_embed_documents_empty_list_makes_no_calls():
    sdk = FakeSDK()
    client = CohereClient(sdk=sdk)
    assert client.embed_documents([]) == []
    assert sdk.embed_calls == []


# --- embed_query ----------------------------------------------------------------------

def test_embed_query_uses_search_query_input_type_and_returns_single_vector():
    sdk = FakeSDK()
    sdk.queue_embed(_embed_response([[0.5] * config.EMBED_DIM]))
    client = CohereClient(sdk=sdk)

    vector = client.embed_query("what's the retry backoff")

    assert sdk.embed_calls[0]["input_type"] == "search_query"
    assert vector == [0.5] * config.EMBED_DIM


# --- rerank -----------------------------------------------------------------------------

def test_rerank_maps_results_and_passes_through_params():
    sdk = FakeSDK()
    sdk.queue_rerank(_rerank_response([(1, 0.9), (0, 0.2)]))
    client = CohereClient(sdk=sdk)

    results = client.rerank(query="retry backoff", documents=["doc a", "doc b"], top_n=2)

    assert sdk.rerank_calls[0]["query"] == "retry backoff"
    assert sdk.rerank_calls[0]["documents"] == ["doc a", "doc b"]
    assert sdk.rerank_calls[0]["top_n"] == 2
    assert [r.index for r in results] == [1, 0]
    assert [r.relevance_score for r in results] == [0.9, 0.2]


# --- retry behavior --------------------------------------------------------------------

def test_transient_error_is_retried_then_succeeds(sleeps):
    calls, fake_sleep = sleeps
    sdk = FakeSDK()
    sdk.queue_embed(cohere.TooManyRequestsError(body="rate limited"))
    sdk.queue_embed(cohere.ServiceUnavailableError(body="unavailable"))
    sdk.queue_embed(_embed_response([[0.1] * config.EMBED_DIM]))
    client = CohereClient(sdk=sdk, max_retries=3, sleep=fake_sleep)

    vector = client.embed_query("hello")

    assert vector == [0.1] * config.EMBED_DIM
    assert len(sdk.embed_calls) == 3
    assert len(calls) == 2  # slept before the 2nd and 3rd attempts


def test_retry_backoff_is_exponential(sleeps):
    calls, fake_sleep = sleeps
    sdk = FakeSDK()
    sdk.queue_embed(cohere.TooManyRequestsError(body="rate limited"))
    sdk.queue_embed(cohere.TooManyRequestsError(body="rate limited"))
    sdk.queue_embed(_embed_response([[0.1] * config.EMBED_DIM]))
    client = CohereClient(sdk=sdk, max_retries=3, backoff_seconds=1.0, sleep=fake_sleep)

    client.embed_query("hello")

    assert calls == [1.0, 2.0]


def test_exhausting_retries_reraises_the_transient_error(sleeps):
    _, fake_sleep = sleeps
    sdk = FakeSDK()
    for _ in range(4):
        sdk.queue_embed(cohere.TooManyRequestsError(body="rate limited"))
    client = CohereClient(sdk=sdk, max_retries=3, sleep=fake_sleep)

    with pytest.raises(cohere.TooManyRequestsError):
        client.embed_query("hello")

    assert len(sdk.embed_calls) == 4  # 1 initial + 3 retries


def test_non_transient_error_is_not_retried(sleeps):
    calls, fake_sleep = sleeps
    sdk = FakeSDK()
    sdk.queue_embed(cohere.BadRequestError(body="bad request"))
    client = CohereClient(sdk=sdk, max_retries=3, sleep=fake_sleep)

    with pytest.raises(cohere.BadRequestError):
        client.embed_query("hello")

    assert len(sdk.embed_calls) == 1
    assert calls == []


def test_rerank_also_retries_on_transient_error(sleeps):
    _, fake_sleep = sleeps
    sdk = FakeSDK()
    sdk.queue_rerank(cohere.TooManyRequestsError(body="rate limited"))
    sdk.queue_rerank(_rerank_response([(0, 0.5)]))
    client = CohereClient(sdk=sdk, max_retries=3, sleep=fake_sleep)

    results = client.rerank(query="q", documents=["d"], top_n=1)

    assert len(sdk.rerank_calls) == 2
    assert results[0].relevance_score == 0.5
