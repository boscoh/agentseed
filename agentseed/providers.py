"""Provider plumbing for the agent.

Builds a Pydantic AI model or embedding model for each supported service and
validates AWS credentials, including the friendly ``aws sso login`` hint for an
expired session.
"""

import configparser
import hashlib
import json
import logging
import os
from datetime import UTC, datetime, timezone
from functools import cache, lru_cache
from http import HTTPStatus
from pathlib import Path
from typing import Any, NamedTuple

import boto3
from botocore.exceptions import TokenRetrievalError
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/v1"


def _log_bedrock_request(http_response, **_: Any) -> None:
    """Log a Bedrock call like httpx does for the HTTP-based providers.

    botocore has no equivalent of httpx's ``HTTP Request: POST ...`` INFO line,
    so this is hooked onto the client's ``after-call`` event.
    """
    status = http_response.status_code
    try:
        reason = HTTPStatus(status).phrase
    except ValueError:
        reason = ""
    logger.info(f'HTTP Request: POST {http_response.url} "HTTP/1.1 {status} {reason}"')


@lru_cache
def load_config() -> dict[str, Any]:
    """Load and return the models configuration from models.json.

    :return: Configuration dictionary with chat_models and embed_models.
    """
    config_path = Path(__file__).parent / "models.json"
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"Could not load models config '{config_path}': {e}") from e
    logger.info(f"Loaded selectable models from '{config_path}'")
    return config


class UnknownModelError(ValueError):
    """Raised for a provider/model that is not listed in models.json."""


class ModelRef(NamedTuple):
    """A chat model listed in models.json; also the key for per-model server state."""

    service: str
    model: str

    def __str__(self) -> str:
        """Return the ``service:model`` form used in logs and reply metadata."""
        return f"{self.service}:{self.model}"


class ModelStatus(BaseModel):
    """The latest health check (probe or real request) for one model."""

    service: str
    model: str
    live: bool
    error: str | None = None
    latency_ms: int | None = None
    checked_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def of(cls, chat_model: ModelRef, **fields: Any) -> "ModelStatus":
        """Build a status for ``chat_model`` with the given fields."""
        return cls(**chat_model._asdict(), **fields)

    @property
    def chat_model(self) -> ModelRef:
        """Return the model this status describes."""
        return ModelRef(self.service, self.model)

    def is_fresh(self, ttl_s: float) -> bool:
        """Return True if live and checked within ``ttl_s`` seconds."""
        age = (datetime.now(UTC) - self.checked_at).total_seconds()
        return self.live and age < ttl_s


def first_line(e: BaseException) -> str:
    """Return a one-line error message for status records."""
    text = str(e).strip()
    return text.splitlines()[0] if text else type(e).__name__


def chat_model_options() -> list[ModelRef]:
    """Return every selectable chat model from models.json.

    :return: ``ModelRef`` per model, in config order.
    """
    options = []
    for service, models in load_config().get("chat_models", {}).items():
        for model in models if isinstance(models, list) else [models]:
            options.append(ModelRef(service, model))
    return options


def resolve_chat_model(
    service: str | None = None, model: str | None = None
) -> ModelRef:
    """Fill in defaults for a provider/model and check it is configured.

    This is the one place chat defaults are decided, all from models.json order:

    - no service: the first provider listed (the server default);
    - no model: that provider's first model.

    :param service: Provider name, or None for the default provider.
    :param model: Model name, or None for the provider's first model.
    :return: A ``ModelRef`` guaranteed to be listed in models.json.
    :raises UnknownModelError: If the chat_model (or provider) is not configured.
    """
    chat_models = chat_model_options()
    if not chat_models:
        raise UnknownModelError("No chat models configured in models.json")
    service = service or chat_models[0].service
    if model is None:
        model = next((m.model for m in chat_models if m.service == service), None)
        if model is None:
            raise UnknownModelError(f"unknown provider: {service}")
    chat_model = ModelRef(service, model)
    if chat_model not in chat_models:
        raise UnknownModelError(f"unknown provider/model: {chat_model}")
    return chat_model


def require_env(name: str) -> str:
    """Return an API key from the environment or raise a friendly error.

    :param name: Environment variable name.
    :return: The non-empty value.
    :raises ValueError: If the variable is unset or empty.
    """
    value = os.getenv(name)
    if not value:
        raise ValueError(
            f"{name} environment variable is not set. "
            f"Please set {name} in your .env file or environment variables."
        )
    return value


def build_provider(service: str) -> Any:
    """Return a Pydantic AI provider with this service's credentials or base URL.

    Shared by ``build_model`` and ``build_embedding_model``. Provider packages
    are imported lazily so a missing optional dependency only affects the
    service that needs it.

    :param service: One of ``openai``, ``anthropic``, ``groq``, ``ollama``, ``bedrock``.
    :return: A configured provider instance.
    :raises ValueError: If the service is unknown or its credentials are missing.
    """
    if service == "openai":
        from pydantic_ai.providers.openai import OpenAIProvider

        return OpenAIProvider(api_key=require_env("OPENAI_API_KEY"))
    if service == "anthropic":
        from pydantic_ai.providers.anthropic import AnthropicProvider

        return AnthropicProvider(api_key=require_env("ANTHROPIC_API_KEY"))
    if service == "groq":
        from pydantic_ai.providers.groq import GroqProvider

        return GroqProvider(api_key=require_env("GROQ_API_KEY"))
    if service == "ollama":
        from pydantic_ai.providers.ollama import OllamaProvider

        return OllamaProvider(
            base_url=os.getenv("OLLAMA_BASE_URL", OLLAMA_DEFAULT_BASE_URL)
        )
    if service == "bedrock":
        from pydantic_ai.providers.bedrock import BedrockProvider

        provider = BedrockProvider(**get_aws_config())
        provider.client.meta.events.register(
            "after-call.bedrock-runtime", _log_bedrock_request
        )
        return provider
    raise ValueError(f"Unknown provider: {service}")


def build_model(service: str, model: str):
    """Return a Pydantic AI chat model for the requested service.

    :param service: One of ``openai``, ``anthropic``, ``groq``, ``ollama``, ``bedrock``.
    :param model: Provider model name.
    :return: A Pydantic AI ``Model`` instance.
    :raises ValueError: If the service is unknown or its credentials are missing.
    """
    Model: Any
    if service == "openai":
        from pydantic_ai.models.openai import OpenAIChatModel as Model
    elif service == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModel as Model
    elif service == "groq":
        from pydantic_ai.models.groq import GroqModel as Model
    elif service == "ollama":
        from pydantic_ai.models.ollama import OllamaModel as Model
    elif service == "bedrock":
        from pydantic_ai.models.bedrock import BedrockConverseModel as Model
    else:
        raise ValueError(f"Unknown chat client type: {service}")
    return Model(model, provider=build_provider(service))


async def check_model(chat_model: ModelRef, timeout: float = 20.0) -> ModelStatus:
    """Probe one provider/model with a tiny live request.

    Sends a one-word prompt capped at a few output tokens, so this exercises
    credentials, network access and model access at negligible cost.

    :param chat_model: Chat model to probe.
    :param timeout: Seconds before the probe is reported as failed.
    :return: The probe result.
    """
    import asyncio
    import time

    from pydantic_ai.direct import model_request
    from pydantic_ai.messages import ModelRequest
    from pydantic_ai.settings import ModelSettings

    start = time.perf_counter()
    error: str | None = None
    try:
        await asyncio.wait_for(
            model_request(
                # Building can block (Bedrock validates AWS credentials over
                # the network), so keep it off the event loop.
                await asyncio.to_thread(build_model, *chat_model),
                [ModelRequest.user_text_prompt("ping")],
                model_settings=ModelSettings(max_tokens=16, timeout=timeout),
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        error = f"timed out after {timeout:g}s"
    except Exception as e:
        error = first_line(e)
    return ModelStatus.of(
        chat_model,
        live=error is None,
        error=error,
        latency_ms=round((time.perf_counter() - start) * 1000),
    )


async def check_models() -> list[ModelStatus]:
    """Probe each provider's default model concurrently.

    :return: One ``check_model`` result per provider, in config order.
    """
    import asyncio

    services = dict.fromkeys(p.service for p in chat_model_options())
    chat_models = [resolve_chat_model(service) for service in services]
    return list(await asyncio.gather(*(check_model(m) for m in chat_models)))


def build_embedding_model(service: str, model: str):
    """Return a Pydantic AI embedding model for the requested service.

    Uses the same ``build_provider`` as the chat models.

    :param service: One of ``openai``, ``ollama``, ``bedrock``.
    :param model: Provider embedding model name.
    :return: A Pydantic AI ``EmbeddingModel`` instance.
    :raises NotImplementedError: For services without embedding support.
    :raises ValueError: If the service is unknown or its credentials are missing.
    """
    Model: Any
    if service in ("openai", "ollama"):
        # Ollama serves an OpenAI-compatible embeddings endpoint.
        from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel as Model
    elif service == "bedrock":
        from pydantic_ai.embeddings.bedrock import BedrockEmbeddingModel as Model
    elif service in ("anthropic", "groq"):
        raise NotImplementedError(
            f"{service} does not support text embeddings. Use openai, ollama, or bedrock."
        )
    else:
        raise ValueError(f"Unknown embedding service: {service}")
    return Model(model, provider=build_provider(service))


async def embed(service: str, model: str, text: str) -> list[float]:
    """Embed text with the given service and model.

    :param service: One of ``openai``, ``ollama``, ``bedrock``.
    :param model: Provider embedding model name.
    :param text: Text to embed.
    :return: Embedding vector.
    :raises NotImplementedError: For services without embedding support.
    :raises ValueError: If the service is unknown or its credentials are missing.
    """
    embedding_model = build_embedding_model(service, model)
    result = await embedding_model.embed(text, input_type="document")
    return list(result.embeddings[0])


def _sso_cache_key_for_profile(home: Path, profile_name: str) -> str | None:
    """Return the SSO cache key configured for a profile.

    A profile is SSO-based when it sets ``sso_session`` or, for legacy configs,
    ``sso_start_url``. Botocore keys the cached token by the session name when
    present, otherwise by the start URL.

    :param home: User home directory containing ``.aws/config``.
    :param profile_name: Profile name, where ``default`` is the unnamed profile.
    :return: Cache key string, or None if the profile is not SSO-based.
    """
    config_path = home / ".aws" / "config"
    if not config_path.exists():
        return None

    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    section = "default" if profile_name == "default" else f"profile {profile_name}"
    if not cfg.has_section(section):
        return None

    session_name = cfg.get(section, "sso_session", fallback=None)
    start_url = cfg.get(section, "sso_start_url", fallback=None)
    return session_name or start_url


def _expired_sso_message(home: Path, profile_name: str) -> str | None:
    """Return a login hint when a profile's cached SSO token is missing or expired.

    Checking the cache before calling AWS means an expired session produces the
    ``aws sso login`` command instead of a raw botocore TokenRetrievalError.

    :param home: User home directory containing the SSO token cache.
    :param profile_name: Effective profile name (``default`` when unset).
    :return: Message telling the user how to log in, or None if not applicable.
    """
    cache_key = _sso_cache_key_for_profile(home, profile_name)
    if not cache_key:
        return None

    hint = (
        f"AWS SSO session expired for profile '{profile_name}'. "
        f"Run: aws sso login --profile {profile_name}"
    )
    # botocore names the SSO token cache file after the SHA1 of the cache key.
    # This must match botocore exactly; it is a filename, not a security use.
    cache_file = (
        home
        / ".aws"
        / "sso"
        / "cache"
        / (
            hashlib.sha1(cache_key.encode("utf-8"), usedforsecurity=False).hexdigest()
            + ".json"
        )
    )
    if not cache_file.exists():
        return hint

    try:
        cache_data = json.loads(cache_file.read_text())
        expires_at = datetime.fromisoformat(
            cache_data["expiresAt"].replace("Z", "+00:00")
        )
    except (json.JSONDecodeError, ValueError, KeyError):
        return None

    return hint if expires_at < datetime.now(timezone.utc) else None


@cache
def get_aws_config() -> dict[str, Any]:
    """Return AWS configuration dict for boto3 client initialization.

    Searches for AWS profiles and credentials, validates them, and checks for
    SSO token expiration. Uses ``AWS_PROFILE`` and ``AWS_REGION`` env vars if set.

    :return: Dict with optional ``profile_name`` and ``region_name`` keys, suitable for unpacking into boto3 constructors.
    :raises ValueError: On missing, incomplete, or expired credentials.
    """
    home = Path.home()
    aws_config: dict[str, Any] = {}

    # Discover available profiles from credentials and config files
    available_profiles = set()
    for aws_file in [home / ".aws" / "credentials", home / ".aws" / "config"]:
        if aws_file.exists():
            cfg = configparser.ConfigParser()
            cfg.read(aws_file)
            for section in cfg.sections():
                if section.startswith("sso-session "):
                    continue
                name = section[8:] if section.startswith("profile ") else section
                available_profiles.add(name)

    # Use AWS_PROFILE if it exists, otherwise fall back to default credential chain
    profile_name = os.getenv("AWS_PROFILE")
    if profile_name:
        if profile_name in available_profiles:
            aws_config["profile_name"] = profile_name
        else:
            logger.info(
                f"AWS profile '{profile_name}' not found, using default credential chain"
            )
            os.environ.pop("AWS_PROFILE", None)
            profile_name = None

    region = os.getenv("AWS_REGION")
    if region:
        aws_config["region_name"] = region

    effective_profile = profile_name or "default"

    # Detect a missing or expired SSO token before calling AWS so the user gets
    # the login command rather than a raw botocore TokenRetrievalError.
    sso_expired_msg = _expired_sso_message(home, effective_profile)
    if sso_expired_msg:
        raise ValueError(sso_expired_msg)

    session = boto3.Session(**aws_config)

    try:
        credentials = session.get_credentials()

        if not credentials:
            hint = (
                f"Available profiles: {', '.join(available_profiles)}\n"
                if available_profiles
                else ""
            )
            raise ValueError(
                f"No AWS credentials found.\n{hint}"
                "To configure: aws configure\n"
                "Or set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY environment variables"
            )

        if not credentials.access_key or not credentials.secret_key:
            raise ValueError(
                "Incomplete AWS credentials (missing access key or secret key)"
            )

        session.client("sts").get_caller_identity()
    except TokenRetrievalError as e:
        raise ValueError(
            f"AWS SSO session expired for profile '{effective_profile}'. "
            f"Run: aws sso login --profile {effective_profile}"
        ) from e

    return aws_config
