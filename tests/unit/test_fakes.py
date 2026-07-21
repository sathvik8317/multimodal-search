from mmsearch import config
from mmsearch.clients.fakes import FakeCaptioner, FakeEmbeddingClient, FakeReranker
from mmsearch.clients.protocols import Captioner, EmbeddingClient, EmbedInput, Reranker


# --- FakeEmbeddingClient ------------------------------------------------------------

def test_fake_embedding_client_conforms_to_protocol():
    assert isinstance(FakeEmbeddingClient(), EmbeddingClient)


def test_fake_embed_documents_returns_correct_dim():
    client = FakeEmbeddingClient()
    vectors = client.embed_documents([EmbedInput(text="hello world")])
    assert len(vectors) == 1
    assert len(vectors[0]) == config.COHERE_EMBED_DIM


def test_fake_embed_documents_handles_image_input():
    client = FakeEmbeddingClient()
    vectors = client.embed_documents([EmbedInput(image_bytes=b"\x89PNG fake bytes")])
    assert len(vectors[0]) == config.COHERE_EMBED_DIM


def test_fake_embed_documents_respects_custom_dim():
    client = FakeEmbeddingClient(dim=config.OPENAI_EMBED_DIM)
    vectors = client.embed_documents([EmbedInput(text="hello world")])
    assert len(vectors[0]) == config.OPENAI_EMBED_DIM


def test_fake_embed_documents_is_deterministic():
    client = FakeEmbeddingClient()
    v1 = client.embed_documents([EmbedInput(text="the retry backoff is exponential")])
    v2 = client.embed_documents([EmbedInput(text="the retry backoff is exponential")])
    assert v1 == v2


def test_fake_embed_documents_differs_for_different_input():
    client = FakeEmbeddingClient()
    v1 = client.embed_documents([EmbedInput(text="alpha")])
    v2 = client.embed_documents([EmbedInput(text="beta")])
    assert v1 != v2


def test_fake_embed_query_returns_correct_dim_and_is_deterministic():
    client = FakeEmbeddingClient()
    q1 = client.embed_query("what's the retry backoff")
    q2 = client.embed_query("what's the retry backoff")
    assert len(q1) == config.COHERE_EMBED_DIM
    assert q1 == q2


def test_fake_embed_documents_batch_preserves_order():
    client = FakeEmbeddingClient()
    a = client.embed_documents([EmbedInput(text="alpha")])[0]
    b = client.embed_documents([EmbedInput(text="beta")])[0]
    batch = client.embed_documents([EmbedInput(text="alpha"), EmbedInput(text="beta")])
    assert batch == [a, b]


# --- FakeReranker ---------------------------------------------------------------------

def test_fake_reranker_conforms_to_protocol():
    assert isinstance(FakeReranker(), Reranker)


def test_fake_reranker_ranks_lexical_overlap_higher():
    reranker = FakeReranker()
    results = reranker.rerank(
        query="retry backoff ingest worker",
        documents=[
            "this document discusses unrelated topics like gardening",
            "the retry backoff for the ingest worker is exponential",
        ],
        top_n=2,
    )
    assert results[0].index == 1
    assert results[0].relevance_score > results[1].relevance_score


def test_fake_reranker_respects_top_n():
    reranker = FakeReranker()
    results = reranker.rerank(
        query="a b c",
        documents=["a", "b", "c", "d"],
        top_n=2,
    )
    assert len(results) == 2


def test_fake_reranker_is_deterministic():
    reranker = FakeReranker()
    docs = ["retry backoff", "unrelated text", "ingest worker retry"]
    r1 = reranker.rerank(query="retry", documents=docs, top_n=3)
    r2 = reranker.rerank(query="retry", documents=docs, top_n=3)
    assert r1 == r2


# --- FakeCaptioner ----------------------------------------------------------------------

def test_fake_captioner_conforms_to_protocol():
    assert isinstance(FakeCaptioner(), Captioner)


def test_fake_captioner_returns_nonempty_string():
    captioner = FakeCaptioner()
    caption = captioner.caption(b"\x89PNG fake diagram bytes")
    assert isinstance(caption, str)
    assert caption

def test_fake_captioner_is_deterministic():
    captioner = FakeCaptioner()
    c1 = captioner.caption(b"same bytes")
    c2 = captioner.caption(b"same bytes")
    assert c1 == c2


def test_fake_captioner_differs_for_different_images():
    captioner = FakeCaptioner()
    c1 = captioner.caption(b"image one")
    c2 = captioner.caption(b"image two")
    assert c1 != c2
