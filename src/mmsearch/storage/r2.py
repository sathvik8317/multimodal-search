"""Cloudflare R2 (S3-compatible) storage for uploaded thumbnails.

Curated thumbnails stay local/in-git (see ingest/thumbnails.py); only
thumbnails produced by the /upload endpoint go through this module. The
underlying boto3 S3 client is injectable (``client=``) so this module can be
unit tested without any network or real credentials -- same pattern as
CohereClient/OpenAIClient's ``sdk=``. In production it lazily builds a real
boto3 client from Settings' AWS_* fields.
"""

from __future__ import annotations

from typing import Any

import boto3
from botocore.exceptions import ClientError

from mmsearch.settings import Settings

_MISSING_KEY_ERROR_CODES = {"NoSuchKey", "404"}


def _build_default_client(settings: Settings) -> Any:
    if not settings.aws_access_key_id:
        raise RuntimeError(
            "AWS_ACCESS_KEY_ID not set and no client provided to R2Storage -- "
            "R2 credentials are required to store/serve uploaded thumbnails"
        )
    return boto3.client(
        "s3",
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        endpoint_url=settings.aws_endpoint_url,
        region_name=settings.aws_region or "auto",
    )


class R2Storage:
    def __init__(
        self, bucket: str, *, client: Any = None, settings: Settings | None = None
    ) -> None:
        self._bucket = bucket
        self._client = client if client is not None else _build_default_client(settings or Settings())

    def put_bytes(self, key: str, data: bytes) -> None:
        self._client.put_object(Bucket=self._bucket, Key=key, Body=data)

    def get_bytes(self, key: str) -> bytes:
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=key)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code in _MISSING_KEY_ERROR_CODES:
                raise FileNotFoundError(key) from exc
            raise
        return response["Body"].read()
