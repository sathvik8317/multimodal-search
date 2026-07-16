"""Local VLM captioner (moondream2) for text-less images: standalone diagram
files and scanned (text-layer-empty) PDF pages (PLAN.md Decision 6, §9).

Torch and transformers are imported lazily, inside _ensure_model_loaded --
never at module import time -- so nothing outside a real captioning call
(ingest, or these tests) ever pulls in torch. Two model queries are run per
image (a dense description, then a verbatim transcription of any visible
text) and composed into one content_text string, so FTS and the reranker
have real text to work with even for images with no text layer.

Captions are cached to disk keyed by the image's content hash (sha256), so
re-running ingest on an already-captioned image never re-invokes the GPU.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Callable
from pathlib import Path

from mmsearch import config

_DESCRIBE_PROMPT = "Describe this image in 2-3 sentences."
_TRANSCRIBE_PROMPT = (
    "Transcribe any text visible in this image, verbatim. "
    "If there is no visible text, respond with an empty string."
)


def _compose_caption(description: str, transcribed_text: str) -> str:
    """Combine a dense description and a verbatim text transcription into one
    content_text string. The transcription section is omitted when nothing
    was transcribed (e.g. a purely decorative image with no visible text).
    """
    transcribed_text = transcribed_text.strip()
    if not transcribed_text:
        return description
    return f"{description}\n\nVisible text: {transcribed_text}"


class LocalCaptioner:
    def __init__(
        self,
        cache_dir: Path = config.CAPTION_CACHE_DIR,
        model_id: str = "vikhyatk/moondream2",
        model_revision: str | None = None,
        _query_fn: Callable[[bytes], str] | None = None,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._model_id = model_id
        self._model_revision = model_revision
        self._query_fn = _query_fn  # test-only injection point; bypasses the real model entirely
        self._model = None  # lazily loaded on first uncached real caption() call

    def caption(self, image_bytes: bytes) -> str:
        cache_path = self._cache_path(image_bytes)
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8")

        text = self._query(image_bytes)
        cache_path.write_text(text, encoding="utf-8")
        return text

    def _cache_path(self, image_bytes: bytes) -> Path:
        digest = hashlib.sha256(image_bytes).hexdigest()
        return self._cache_dir / f"{digest}.txt"

    def _query(self, image_bytes: bytes) -> str:
        if self._query_fn is not None:
            return self._query_fn(image_bytes)
        return self._query_real_model(image_bytes)

    def _query_real_model(self, image_bytes: bytes) -> str:
        self._ensure_model_loaded()
        image = _to_pil_image(image_bytes)
        description = self._model.query(image, _DESCRIBE_PROMPT)["answer"]
        transcribed = self._model.query(image, _TRANSCRIBE_PROMPT)["answer"]
        return _compose_caption(description, transcribed)

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM

        kwargs = {"trust_remote_code": True, "torch_dtype": torch.float16}
        if self._model_revision is not None:
            kwargs["revision"] = self._model_revision

        model = AutoModelForCausalLM.from_pretrained(self._model_id, **kwargs)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        self._model = model


def _to_pil_image(image_bytes: bytes):
    from PIL import Image

    return Image.open(io.BytesIO(image_bytes)).convert("RGB")
