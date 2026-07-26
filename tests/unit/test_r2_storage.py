import io

import pytest
from botocore.exceptions import ClientError

from mmsearch.settings import Settings
from mmsearch.storage.r2 import R2Storage


class _FakeS3Client:
    """In-memory stand-in for a boto3 S3 client, mirroring the sdk= injection
    pattern used by CohereClient/OpenAIClient -- no network, no moto."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def put_object(self, Bucket: str, Key: str, Body: bytes) -> None:
        self.objects[(Bucket, Key)] = Body

    def get_object(self, Bucket: str, Key: str) -> dict:
        if (Bucket, Key) not in self.objects:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "not found"}}, "GetObject"
            )
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}


# --- put_bytes / get_bytes ------------------------------------------------------------------

def test_put_bytes_stores_object_under_bucket_and_key():
    client = _FakeS3Client()
    storage = R2Storage("my-bucket", client=client)

    storage.put_bytes("uploads/a.png", b"png-bytes")

    assert client.objects[("my-bucket", "uploads/a.png")] == b"png-bytes"


def test_get_bytes_returns_stored_object():
    client = _FakeS3Client()
    storage = R2Storage("my-bucket", client=client)
    storage.put_bytes("uploads/a.png", b"png-bytes")

    assert storage.get_bytes("uploads/a.png") == b"png-bytes"


def test_get_bytes_raises_file_not_found_for_missing_key():
    client = _FakeS3Client()
    storage = R2Storage("my-bucket", client=client)

    with pytest.raises(FileNotFoundError):
        storage.get_bytes("uploads/missing.png")


def test_get_bytes_reraises_non_missing_client_errors():
    client = _FakeS3Client()

    def _raise_access_denied(**kwargs):
        raise ClientError({"Error": {"Code": "AccessDenied", "Message": "nope"}}, "GetObject")

    client.get_object = _raise_access_denied
    storage = R2Storage("my-bucket", client=client)

    with pytest.raises(ClientError):
        storage.get_bytes("uploads/a.png")


# --- default client construction -------------------------------------------------------------

def test_raises_clear_error_when_no_client_and_no_credentials():
    settings = Settings(_env_file=None, aws_access_key_id=None)

    with pytest.raises(RuntimeError, match="AWS_ACCESS_KEY_ID"):
        R2Storage("my-bucket", settings=settings)
