import pytest
from pydantic import ValidationError

from mmsearch.settings import Settings, get_settings


# --- basic env reads ---------------------------------------------------------------------

def test_reads_api_key_from_mmsearch_prefixed_env_var(monkeypatch):
    monkeypatch.setenv("MMSEARCH_API_KEY", "secret123")

    settings = Settings(_env_file=None)

    assert settings.api_key == "secret123"


def test_reads_cohere_api_key_from_unprefixed_env_var(monkeypatch):
    monkeypatch.setenv("MMSEARCH_API_KEY", "secret123")
    monkeypatch.setenv("COHERE_API_KEY", "cohere-secret")

    settings = Settings(_env_file=None)

    assert settings.cohere_api_key == "cohere-secret"


def test_cohere_api_key_defaults_to_none_when_unset(monkeypatch):
    monkeypatch.setenv("MMSEARCH_API_KEY", "secret123")
    monkeypatch.delenv("COHERE_API_KEY", raising=False)

    settings = Settings(_env_file=None)

    assert settings.cohere_api_key is None


def test_reads_openai_api_key_from_unprefixed_env_var(monkeypatch):
    monkeypatch.setenv("MMSEARCH_API_KEY", "secret123")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")

    settings = Settings(_env_file=None)

    assert settings.openai_api_key == "openai-secret"


def test_openai_api_key_defaults_to_none_when_unset(monkeypatch):
    monkeypatch.setenv("MMSEARCH_API_KEY", "secret123")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    settings = Settings(_env_file=None)

    assert settings.openai_api_key is None


# --- .env file ----------------------------------------------------------------------------

def test_dotenv_file_is_honored_cwd_relative(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MMSEARCH_API_KEY", raising=False)
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (tmp_path / ".env").write_text(
        "MMSEARCH_API_KEY=from-dotenv\n"
        "COHERE_API_KEY=cohere-from-dotenv\n"
        "OPENAI_API_KEY=openai-from-dotenv\n"
    )

    settings = Settings()

    assert settings.api_key == "from-dotenv"
    assert settings.cohere_api_key == "cohere-from-dotenv"
    assert settings.openai_api_key == "openai-from-dotenv"


def test_real_env_var_overrides_dotenv_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MMSEARCH_API_KEY", "from-real-env")
    (tmp_path / ".env").write_text("MMSEARCH_API_KEY=from-dotenv\n")

    settings = Settings()

    assert settings.api_key == "from-real-env"


# --- Settings() itself never enforces the gate-key policy ------------------------------------
#
# Constructing a Settings() must always succeed regardless of api_key/env, because
# clients/cohere.py constructs a bare Settings() to read cohere_api_key -- unrelated
# to whether the /search gate secret happens to be configured. The policy (dev
# requires a gate key) lives in get_settings() instead; see below.

def test_settings_construction_never_raises_for_missing_api_key(monkeypatch):
    monkeypatch.delenv("MMSEARCH_API_KEY", raising=False)

    settings = Settings(_env_file=None)  # must not raise

    assert settings.api_key is None


def test_env_rejects_value_outside_allowed_literals(monkeypatch):
    monkeypatch.setenv("MMSEARCH_API_KEY", "secret123")

    with pytest.raises(ValidationError):
        Settings(_env_file=None, env="qa")


def test_env_accepts_staging_without_requiring_api_key(monkeypatch):
    monkeypatch.delenv("MMSEARCH_API_KEY", raising=False)

    settings = Settings(_env_file=None, env="staging")

    assert settings.env == "staging"
    assert settings.api_key is None


# --- get_settings() enforces the gate-key policy ----------------------------------------------

def test_get_settings_raises_clear_runtime_error_when_api_key_missing_in_dev(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no .env file here
    monkeypatch.delenv("MMSEARCH_API_KEY", raising=False)
    get_settings.cache_clear()

    try:
        with pytest.raises(RuntimeError, match="MMSEARCH_API_KEY"):
            get_settings()
    finally:
        get_settings.cache_clear()


def test_get_settings_succeeds_when_api_key_present(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MMSEARCH_API_KEY", "secret123")
    get_settings.cache_clear()

    try:
        settings = get_settings()
        assert settings.api_key == "secret123"
    finally:
        get_settings.cache_clear()


# --- allowed_origins ------------------------------------------------------------------------

def test_allowed_origins_defaults_to_localhost_vite_ports(monkeypatch):
    monkeypatch.setenv("MMSEARCH_API_KEY", "secret123")

    settings = Settings(_env_file=None)

    assert settings.allowed_origins == [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


def test_allowed_origins_parses_comma_separated_env_string(monkeypatch):
    monkeypatch.setenv("MMSEARCH_API_KEY", "secret123")
    monkeypatch.setenv("MMSEARCH_ALLOWED_ORIGINS", "http://a.test,http://b.test")

    settings = Settings(_env_file=None)

    assert settings.allowed_origins == ["http://a.test", "http://b.test"]


# --- rate limit knobs -----------------------------------------------------------------------

def test_rate_limit_knobs_have_sane_positive_defaults(monkeypatch):
    monkeypatch.setenv("MMSEARCH_API_KEY", "secret123")

    settings = Settings(_env_file=None)

    assert settings.rate_limit_max > 0
    assert settings.rate_limit_window > 0


# --- get_settings caching --------------------------------------------------------------------

def test_get_settings_is_cached(monkeypatch):
    monkeypatch.setenv("MMSEARCH_API_KEY", "secret123")
    get_settings.cache_clear()

    first = get_settings()
    second = get_settings()

    assert first is second
    get_settings.cache_clear()
