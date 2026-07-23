"""Hosted VLM captioner (OpenAI vision) for the server /upload path only.

Local ingestion keeps using LocalCaptioner (moondream2); this exists because
the deployed Render instance has no GPU/torch and captioning at upload time
must still work. Reuses LocalCaptioner's prompts and _compose_caption so
caption style/format stays consistent between the two paths. The underlying
OpenAI SDK client is injectable (``sdk=``), same pattern as OpenAIClient.
"""

from __future__ import annotations

import base64
import time
from collections.abc import Callable
from typing import Any

import openai

from mmsearch import config
from mmsearch.clients.captioner_local import (
    _DESCRIBE_PROMPT,
    _TRANSCRIBE_PROMPT,
    _compose_caption,
)
from mmsearch.settings import Settings

_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BACKOFF_SECONDS = 1.0
_MAX_OUTPUT_TOKENS = 300

# Same transient-error policy as OpenAIClient (see clients/openai.py).
_TRANSIENT_ERRORS = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


def _build_default_sdk(api_key: str | None) -> openai.OpenAI:
    key = api_key or Settings().openai_api_key
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY not set and no api_key/sdk provided to ApiCaptioner"
        )
    return openai.OpenAI(api_key=key)


def _to_data_uri(image_bytes: bytes) -> str:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"


class ApiCaptioner:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        sdk: Any = None,
        model: str = config.OPENAI_VISION_MODEL,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        backoff_seconds: float = _DEFAULT_BACKOFF_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._sdk = sdk if sdk is not None else _build_default_sdk(api_key)
        self._model = model
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._sleep = sleep

    def caption(self, image_bytes: bytes) -> str:
        description = self._query(image_bytes, _DESCRIBE_PROMPT)
        transcribed = self._query(image_bytes, _TRANSCRIBE_PROMPT)
        return _compose_caption(description, transcribed)

    def _query(self, image_bytes: bytes, prompt: str) -> str:
        response = self._call_with_retry(
            lambda: self._sdk.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": _to_data_uri(image_bytes)}},
                        ],
                    }
                ],
                max_tokens=_MAX_OUTPUT_TOKENS,
            )
        )
        return response.choices[0].message.content or ""

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
