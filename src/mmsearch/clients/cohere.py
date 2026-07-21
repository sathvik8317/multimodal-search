"""Real Cohere client: Embed v4 (documents + query) and Rerank v3.

The underlying SDK object is injectable (``sdk=``) so this module can be unit
tested without ever touching the network. In production it lazily builds a
real ``cohere.ClientV2`` from ``COHERE_API_KEY``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import cohere
from cohere.types import EmbedImageUrl, ImageUrlEmbedContent, TextEmbedContent
from cohere.types import EmbedInput as SDKEmbedInput

from mmsearch import config
from mmsearch.clients.protocols import EmbedInput, RerankResult
from mmsearch.settings import Settings

_EMBED_BATCH_SIZE = 96
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BACKOFF_SECONDS = 1.0

# Errors worth retrying: rate limits and transient server-side failures.
_TRANSIENT_ERRORS = (
    cohere.TooManyRequestsError,
    cohere.ServiceUnavailableError,
    cohere.GatewayTimeoutError,
    cohere.InternalServerError,
)


def _build_default_sdk(api_key: str | None) -> cohere.ClientV2:
    # Settings(), not get_settings(): a bare Settings() never enforces the
    # unrelated /search gate-key policy (see settings.py), and reads .env the
    # same cwd-relative way load_dotenv() used to.
    key = api_key or Settings().cohere_api_key
    if not key:
        raise RuntimeError(
            "COHERE_API_KEY not set and no api_key/sdk provided to CohereClient"
        )
    return cohere.ClientV2(api_key=key)


def _to_sdk_embed_input(item: EmbedInput) -> SDKEmbedInput:
    if item.text is not None:
        return SDKEmbedInput(content=[TextEmbedContent(text=item.text)])
    b64 = _b64encode(item.image_bytes)
    return SDKEmbedInput(
        content=[ImageUrlEmbedContent(image_url=EmbedImageUrl(url=f"data:image/png;base64,{b64}"))]
    )


def _b64encode(data: bytes) -> str:
    import base64

    return base64.b64encode(data).decode("ascii")


def _chunk(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class CohereClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        sdk: Any = None,
        embed_batch_size: int = _EMBED_BATCH_SIZE,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        backoff_seconds: float = _DEFAULT_BACKOFF_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._sdk = sdk if sdk is not None else _build_default_sdk(api_key)
        self._embed_batch_size = embed_batch_size
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._sleep = sleep

    def embed_documents(self, items: list[EmbedInput]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for batch in _chunk(items, self._embed_batch_size):
            response = self._call_with_retry(
                lambda batch=batch: self._sdk.embed(
                    model=config.COHERE_EMBED_MODEL,
                    input_type="search_document",
                    inputs=[_to_sdk_embed_input(item) for item in batch],
                    embedding_types=["float"],
                )
            )
            vectors.extend(response.embeddings.float_)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        response = self._call_with_retry(
            lambda: self._sdk.embed(
                model=config.COHERE_EMBED_MODEL,
                input_type="search_query",
                inputs=[_to_sdk_embed_input(EmbedInput(text=text))],
                embedding_types=["float"],
            )
        )
        return response.embeddings.float_[0]

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[RerankResult]:
        response = self._call_with_retry(
            lambda: self._sdk.rerank(
                model=config.RERANK_MODEL,
                query=query,
                documents=documents,
                top_n=top_n,
            )
        )
        return [
            RerankResult(index=item.index, relevance_score=item.relevance_score)
            for item in response.results
        ]

    def _call_with_retry(self, fn: Callable[[], Any]) -> Any:
        attempt = 0
        while True:
            try:
                return fn()
            except _TRANSIENT_ERRORS:
                attempt += 1
                if attempt > self._max_retries:
                    raise
                self._sleep(self._backoff_seconds * (2 ** (attempt - 1)))
