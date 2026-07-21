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
    # Declared for future use; only "dev" has behavior attached (see get_settings).
    env: Literal["dev", "staging", "prod"] = "dev"

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _split_comma_separated(cls, value: object) -> object:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    if settings.env == "dev" and not settings.api_key:
        raise RuntimeError(
            "MMSEARCH_API_KEY is not set. Generate one and add it to .env, e.g.\n"
            '  MMSEARCH_API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")'
        )
    return settings
