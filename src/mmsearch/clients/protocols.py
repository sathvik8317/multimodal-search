from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class EmbedInput:
    """One item to embed as a document. Exactly one of text/image_bytes is set."""

    text: str | None = None
    image_bytes: bytes | None = None

    def __post_init__(self) -> None:
        if (self.text is None) == (self.image_bytes is None):
            raise ValueError("exactly one of text or image_bytes must be set")


@dataclass(frozen=True)
class RerankResult:
    index: int
    relevance_score: float


@runtime_checkable
class EmbeddingClient(Protocol):
    def embed_documents(self, items: list[EmbedInput]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, documents: list[str], top_n: int) -> list[RerankResult]: ...


@runtime_checkable
class Captioner(Protocol):
    def caption(self, image_bytes: bytes) -> str: ...


@dataclass(frozen=True)
class Embedders:
    """The two embedding clients ingest routes to, by input kind.

    image: embeds page/diagram raster bytes (Cohere Embed v4 -- the only
    provider with a unified image+text space; see EMBEDDING_MIGRATION_PLAN.md).
    text: embeds table/code/caption text (OpenAI text-embedding-3-small).
    """

    image: EmbeddingClient
    text: EmbeddingClient
