"""Real OpenAI client: text-embedding-3-small (documents + query).

Text only -- OpenAI's embeddings API has no image input, unlike Cohere Embed
v4. The underlying SDK object is injectable (``sdk=``) so this module can be
unit tested without ever touching the network. In production it lazily
builds a real ``openai.OpenAI`` client from ``OPENAI_API_KEY``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import openai

from mmsearch import config
from mmsearch.clients.protocols import EmbedInput
from mmsearch.settings import Settings

_EMBED_BATCH_SIZE = 96
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BACKOFF_SECONDS = 1.0

# Errors worth retrying: rate limits and transient server-side failures.
# Names/hierarchy confirmed against the installed openai>=1.0 SDK, not assumed
# from memory (see EMBEDDING_MIGRATION_PLAN.md verification step 1).
_TRANSIENT_ERRORS = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


def _build_default_sdk(api_key: str | None) -> openai.OpenAI:
    # Settings(), not get_settings(): see clients/cohere.py's identical comment
    # -- a bare Settings() never enforces the unrelated /search gate-key policy.
    key = api_key or Settings().openai_api_key
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY not set and no api_key/sdk provided to OpenAIClient"
        )
    return openai.OpenAI(api_key=key)


def _chunk(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class OpenAIClient:
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
        for item in items:
            if item.image_bytes is not None:
                raise ValueError("OpenAIClient only embeds text; got an image EmbedInput")

        vectors: list[list[float]] = []
        for batch in _chunk(items, self._embed_batch_size):
            response = self._call_with_retry(
                lambda batch=batch: self._sdk.embeddings.create(
                    model=config.OPENAI_EMBED_MODEL,
                    input=[item.text for item in batch],
                )
            )
            vectors.extend(d.embedding for d in response.data)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([EmbedInput(text=text)])[0]

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


def demo() -> None:
    """Self-check: OpenAIClient rejects image input (text-only assumption)."""
    from mmsearch.clients.protocols import EmbedInput as _EmbedInput

    client = OpenAIClient(sdk=object())  # sdk unused: raises before any call
    try:
        client.embed_documents([_EmbedInput(image_bytes=b"\x89PNG")])
    except ValueError as exc:
        assert "only embeds text" in str(exc)
    else:
        raise AssertionError("expected ValueError for image input")
    print("ok: OpenAIClient rejects image EmbedInput")


if __name__ == "__main__":
    demo()
