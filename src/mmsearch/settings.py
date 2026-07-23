"""Typed application settings, loaded from environment variables and .env.

``Settings`` is a pure data model: constructing it never fails based on which
fields are populated. The "a gate secret is required in dev" policy lives in
``get_settings()`` instead, because ``clients/cohere.py`` needs to construct a
bare ``Settings()`` to read ``cohere_api_key`` -- unrelated to whether the
``/search`` gate secret (``api_key``) happens to be configured -- and dropping
``load_dotenv()`` means it can no longer fall back to reading ``os.environ``
directly (nothing mutates it process-wide anymore; pydantic-settings parses
``.env`` internally, scoped to the Settings instance).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MMSEARCH_", env_file=".env", extra="ignore")

    # Unprefixed and unenforced here, mirroring the existing convention:
    # CohereClient raises a clear RuntimeError lazily, only when it actually
    # needs the key (see clients/cohere.py).
    cohere_api_key: str | None = Field(default=None, validation_alias="COHERE_API_KEY")
    # Same convention as cohere_api_key above; OpenAIClient raises lazily too.
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    # The shared secret that gates /search and /thumbnails. Enforced by
    # get_settings(), not here -- see module docstring.
    api_key: str | None = None
    # NoDecode: pydantic-settings otherwise tries to JSON-parse list-typed env
    # values before our validator runs, which rejects a plain comma-separated
    # string with a SettingsError.
    allowed_origins: Annotated[list[str], NoDecode] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    rate_limit_max: int = 20
    rate_limit_window: float = 60.0
    # Upload-specific limit, deliberately separate from rate_limit_max/window
    # (see api/deps.py) -- ingestion is far more expensive per request than a
    # search, so it gets its own, stricter budget.
    upload_rate_limit_max: int = 5
    upload_rate_limit_window: float = 60.0
    # Declared for future use; only "dev" has behavior attached (see get_settings).
    env: Literal["dev", "staging", "prod"] = "dev"

    # None means "use the local config.LANCEDB_URI path" (db.open_table's
    # default). Set to an s3://bucket/... URI to point at Cloudflare R2.
    lancedb_uri: str | None = None
    # Bucket uploaded thumbnails are stored in (storage/r2.py). Curated
    # thumbnails always stay local/in-git regardless of this setting.
    r2_bucket: str | None = None

    # Unprefixed, same convention as cohere_api_key/openai_api_key: these are
    # the standard AWS SDK env var names, shared with boto3's own credential
    # chain (see storage/r2.py) so one set of Render env vars configures both
    # LanceDB's object store and the thumbnail uploader.
    aws_access_key_id: str | None = Field(default=None, validation_alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str | None = Field(default=None, validation_alias="AWS_SECRET_ACCESS_KEY")
    aws_endpoint_url: str | None = Field(default=None, validation_alias="AWS_ENDPOINT_URL")
    aws_region: str | None = Field(default=None, validation_alias="AWS_REGION")

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _split_comma_separated(cls, value: object) -> object:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    def r2_storage_options(self) -> dict[str, str] | None:
        """Build LanceDB's ``storage_options`` dict for R2, or None for local dev.

        None (no aws_access_key_id configured) tells db.open_table() to connect
        with no storage_options at all, which is what a local filesystem uri
        needs -- there's nothing to authenticate.
        """
        if not self.aws_access_key_id:
            return None
        return {
            "aws_access_key_id": self.aws_access_key_id,
            "aws_secret_access_key": self.aws_secret_access_key or "",
            "aws_endpoint": self.aws_endpoint_url or "",
            "aws_region": self.aws_region or "",
        }


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    if settings.env == "dev" and not settings.api_key:
        raise RuntimeError(
            "MMSEARCH_API_KEY is not set. Generate one and add it to .env, e.g.\n"
            '  MMSEARCH_API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")'
        )
    return settings
