import pytest

from mmsearch.clients.protocols import (
    Captioner,
    Embedders,
    EmbeddingClient,
    EmbedInput,
    Reranker,
    RerankResult,
)


# --- EmbedInput -----------------------------------------------------------------

def test_embed_input_text_only_is_valid():
    item = EmbedInput(text="def f(): pass")
    assert item.text == "def f(): pass"
    assert item.image_bytes is None


def test_embed_input_image_only_is_valid():
    item = EmbedInput(image_bytes=b"\x89PNG...")
    assert item.image_bytes == b"\x89PNG..."
    assert item.text is None


def test_embed_input_rejects_both_set():
    with pytest.raises(ValueError, match="exactly one"):
        EmbedInput(text="hello", image_bytes=b"data")


def test_embed_input_rejects_neither_set():
    with pytest.raises(ValueError, match="exactly one"):
        EmbedInput()


# --- RerankResult -----------------------------------------------------------------

def test_rerank_result_fields():
    result = RerankResult(index=2, relevance_score=0.93)
    assert result.index == 2
    assert result.relevance_score == 0.93


# --- Protocol conformance (runtime_checkable) --------------------------------------

def test_embedding_client_protocol_accepts_conforming_object():
    class ConformingClient:
        def embed_documents(self, items: list[EmbedInput]) -> list[list[float]]:
            return [[0.0]]

        def embed_query(self, text: str) -> list[float]:
            return [0.0]

    assert isinstance(ConformingClient(), EmbeddingClient)


def test_embedding_client_protocol_rejects_partial_object():
    class MissingEmbedQuery:
        def embed_documents(self, items: list[EmbedInput]) -> list[list[float]]:
            return [[0.0]]

    assert not isinstance(MissingEmbedQuery(), EmbeddingClient)


def test_reranker_protocol_accepts_conforming_object():
    class ConformingReranker:
        def rerank(self, query: str, documents: list[str], top_n: int) -> list[RerankResult]:
            return [RerankResult(index=0, relevance_score=1.0)]

    assert isinstance(ConformingReranker(), Reranker)


def test_reranker_protocol_rejects_partial_object():
    class NotAReranker:
        pass

    assert not isinstance(NotAReranker(), Reranker)


def test_captioner_protocol_accepts_conforming_object():
    class ConformingCaptioner:
        def caption(self, image_bytes: bytes) -> str:
            return "a diagram"

    assert isinstance(ConformingCaptioner(), Captioner)


def test_captioner_protocol_rejects_partial_object():
    class NotACaptioner:
        pass

    assert not isinstance(NotACaptioner(), Captioner)


# --- Embedders --------------------------------------------------------------------

def test_embedders_holds_image_and_text_clients():
    class ConformingClient:
        def embed_documents(self, items: list[EmbedInput]) -> list[list[float]]:
            return [[0.0]]

        def embed_query(self, text: str) -> list[float]:
            return [0.0]

    image_client = ConformingClient()
    text_client = ConformingClient()
    embedders = Embedders(image=image_client, text=text_client)

    assert embedders.image is image_client
    assert embedders.text is text_client
