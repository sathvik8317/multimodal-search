"""Tests for the hosted-VLM captioner (OpenAI vision). The underlying SDK
object is always injected as a fake, same pattern as test_openai_client.py --
these tests never construct a real openai.OpenAI and never touch the
network."""

import httpx
import openai
import pytest

from mmsearch import config
from mmsearch.clients.captioner_api import ApiCaptioner
from mmsearch.clients.protocols import Captioner

_REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def _chat_response(text: str):
    class _Message:
        content = text

    class _Choice:
        message = _Message()

    class _Response:
        choices = [_Choice()]

    return _Response()


def _rate_limit_error() -> openai.RateLimitError:
    return openai.RateLimitError(
        "rate limited", response=httpx.Response(429, request=_REQUEST), body=None
    )


def _bad_request_error() -> openai.BadRequestError:
    return openai.BadRequestError(
        "bad request", response=httpx.Response(400, request=_REQUEST), body=None
    )


class FakeSDK:
    """Records calls; returns canned responses in order; can raise N times."""

    def __init__(self):
        self.calls: list[dict] = []
        self._queue: list = []
        self.chat = self
        self.completions = self

    def queue(self, response_or_exc):
        self._queue.append(response_or_exc)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def sleeps():
    calls: list[float] = []

    def fake_sleep(seconds: float) -> None:
        calls.append(seconds)

    return calls, fake_sleep


# --- protocol conformance --------------------------------------------------------------

def test_api_captioner_conforms_to_captioner_protocol():
    sdk = FakeSDK()
    captioner = ApiCaptioner(sdk=sdk)
    assert isinstance(captioner, Captioner)


def test_construction_without_sdk_or_api_key_raises_clear_error(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        ApiCaptioner()


# --- caption(): two-query composition, mirroring LocalCaptioner ------------------------

def test_caption_sends_two_queries_describe_then_transcribe():
    sdk = FakeSDK()
    sdk.queue(_chat_response("A diagram showing an auth flow."))
    sdk.queue(_chat_response("Client, Auth Server"))
    captioner = ApiCaptioner(sdk=sdk)

    result = captioner.caption(b"fake-png-bytes")

    assert len(sdk.calls) == 2
    assert "Describe" in sdk.calls[0]["messages"][0]["content"][0]["text"]
    assert "Transcribe" in sdk.calls[1]["messages"][0]["content"][0]["text"]
    assert "A diagram showing an auth flow." in result
    assert "Client, Auth Server" in result


def test_caption_omits_transcription_section_when_no_text_visible():
    sdk = FakeSDK()
    sdk.queue(_chat_response("A plain gradient background."))
    sdk.queue(_chat_response(""))
    captioner = ApiCaptioner(sdk=sdk)

    result = captioner.caption(b"fake-png-bytes")

    assert result == "A plain gradient background."


def test_caption_sends_image_as_base64_data_uri():
    import base64

    sdk = FakeSDK()
    sdk.queue(_chat_response("description"))
    sdk.queue(_chat_response(""))
    captioner = ApiCaptioner(sdk=sdk)

    captioner.caption(b"raw-bytes")

    b64 = base64.b64encode(b"raw-bytes").decode("ascii")
    image_url = sdk.calls[0]["messages"][0]["content"][1]["image_url"]["url"]
    assert image_url == f"data:image/png;base64,{b64}"


def test_caption_uses_configured_vision_model():
    sdk = FakeSDK()
    sdk.queue(_chat_response("d"))
    sdk.queue(_chat_response(""))
    captioner = ApiCaptioner(sdk=sdk)

    captioner.caption(b"bytes")

    assert sdk.calls[0]["model"] == config.OPENAI_VISION_MODEL


# --- retry behavior (same policy as OpenAIClient) ---------------------------------------

def test_transient_error_is_retried_then_succeeds(sleeps):
    calls, fake_sleep = sleeps
    sdk = FakeSDK()
    sdk.queue(_rate_limit_error())
    sdk.queue(_chat_response("description"))
    sdk.queue(_chat_response(""))
    captioner = ApiCaptioner(sdk=sdk, max_retries=3, sleep=fake_sleep)

    result = captioner.caption(b"bytes")

    assert result == "description"
    assert len(sdk.calls) == 3
    assert len(calls) == 1


def test_exhausting_retries_reraises_the_transient_error(sleeps):
    _, fake_sleep = sleeps
    sdk = FakeSDK()
    for _ in range(4):
        sdk.queue(_rate_limit_error())
    captioner = ApiCaptioner(sdk=sdk, max_retries=3, sleep=fake_sleep)

    with pytest.raises(openai.RateLimitError):
        captioner.caption(b"bytes")

    assert len(sdk.calls) == 4


def test_non_transient_error_is_not_retried(sleeps):
    calls, fake_sleep = sleeps
    sdk = FakeSDK()
    sdk.queue(_bad_request_error())
    captioner = ApiCaptioner(sdk=sdk, max_retries=3, sleep=fake_sleep)

    with pytest.raises(openai.BadRequestError):
        captioner.caption(b"bytes")

    assert len(sdk.calls) == 1
    assert calls == []
