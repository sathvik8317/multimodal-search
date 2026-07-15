"""Deterministic fakes for EmbeddingClient, Reranker, Captioner.

No network, no GPU. Used by every Phase-1 track to build/test against the
frozen protocols before a real index or real API keys exist.
"""

from __future__ import annotations

import hashlib
import random

from mmsearch import config
from mmsearch.clients.protocols import EmbedInput, RerankResult


def _deterministic_vector(seed_bytes: bytes) -> list[float]:
    digest = hashlib.sha256(seed_bytes).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    vector = [rng.uniform(-1.0, 1.0) for _ in range(config.EMBED_DIM)]
    norm = sum(x * x for x in vector) ** 0.5 or 1.0
    return [x / norm for x in vector]


class FakeEmbeddingClient:
    def embed_documents(self, items: list[EmbedInput]) -> list[list[float]]:
        return [
            _deterministic_vector(item.text.encode() if item.text is not None else item.image_bytes)
            for item in items
        ]

    def embed_query(self, text: str) -> list[float]:
        return _deterministic_vector(text.encode())


class FakeReranker:
    def rerank(self, query: str, documents: list[str], top_n: int) -> list[RerankResult]:
        query_tokens = set(query.lower().split())
        scored = []
        for index, document in enumerate(documents):
            document_tokens = set(document.lower().split())
            overlap = len(query_tokens & document_tokens)
            score = overlap / (len(query_tokens) or 1)
            scored.append(RerankResult(index=index, relevance_score=score))
        scored.sort(key=lambda r: (-r.relevance_score, r.index))
        return scored[:top_n]


class FakeCaptioner:
    def caption(self, image_bytes: bytes) -> str:
        digest = hashlib.sha256(image_bytes).hexdigest()[:8]
        return f"[fake-caption] hash={digest} size={len(image_bytes)}b"
