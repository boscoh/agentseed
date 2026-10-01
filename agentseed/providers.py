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
from datetime import datetime, timezone
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import TokenRetrievalError

logger = logging.getLogger(__name__)

OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/v1"


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


def default_model(service: str) -> str | None:
    """Return the first configured chat model for a service from models.json.

    :param service: One of ``openai``, ``anthropic``, ``groq``, ``ollama``, ``bedrock``.
    :return: Model name, or None if the service has no configured models.
    """
    models = load_config().get("chat_models", {}).get(service, [])
    if isinstance(models, list):
        return models[0] if models else None
    return models


def chat_model_options() -> list[dict[str, str]]:
    """Return every selectable chat provider/model pair from models.json.

    :return: List of ``{"service": ..., "model": ...}`` dicts in config order.
    """
    options = []
    for service, models in load_config().get("chat_models", {}).items():
        for model in models if isinstance(models, list) else [models]:
            options.append({"service": service, "model": model})
    return options


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


def build_model(service: str, model: str):
    """Return a Pydantic AI model for the requested service.

    Provider packages are imported lazily so a missing optional dependency only
    affects the service that needs it.

    :param service: One of ``openai``, ``anthropic``, ``groq``, ``ollama``, ``bedrock``.
    :param model: Provider model name.
    :return: A Pydantic AI ``Model`` instance.
    :raises ValueError: If the service is unknown or its credentials are missing.
    """
    Model: Any
    Provider: Any
    kwargs: dict[str, Any]
    if service == "openai":
        from pydantic_ai.models.openai import OpenAIChatModel as Model
        from pydantic_ai.providers.openai import OpenAIProvider as Provider

        kwargs = {"api_key": require_env("OPENAI_API_KEY")}
    elif service == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModel as Model
        from pydantic_ai.providers.anthropic import AnthropicProvider as Provider

        kwargs = {"api_key": require_env("ANTHROPIC_API_KEY")}
    elif service == "groq":
        from pydantic_ai.models.groq import GroqModel as Model
        from pydantic_ai.providers.groq import GroqProvider as Provider

        kwargs = {"api_key": require_env("GROQ_API_KEY")}
    elif service == "ollama":
        from pydantic_ai.models.ollama import OllamaModel as Model
        from pydantic_ai.providers.ollama import OllamaProvider as Provider

        kwargs = {"base_url": os.getenv("OLLAMA_BASE_URL", OLLAMA_DEFAULT_BASE_URL)}
    elif service == "bedrock":
        from pydantic_ai.models.bedrock import BedrockConverseModel as Model
        from pydantic_ai.providers.bedrock import BedrockProvider as Provider

        kwargs = get_aws_config()
    else:
        raise ValueError(f"Unknown chat client type: {service}")
    return Model(model, provider=Provider(**kwargs))


async def check_model(service: str, model: str, timeout: float = 20.0) -> dict[str, Any]:
    """Probe one provider/model with a tiny live request.

    Sends a one-word prompt capped at a few output tokens, so this exercises
    credentials, network access and model access at negligible cost.

    :param service: Provider name, e.g. ``anthropic``.
    :param model: Provider model name.
    :param timeout: Seconds before the probe is reported as failed.
    :return: ``{"service", "model", "live", "latency_ms", "error"}``.
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
                await asyncio.to_thread(build_model, service, model),
                [ModelRequest.user_text_prompt("ping")],
                model_settings=ModelSettings(max_tokens=16, timeout=timeout),
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        error = f"timed out after {timeout:g}s"
    except Exception as e:
        error = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
    return {
        "service": service,
        "model": model,
        "live": error is None,
        "latency_ms": round((time.perf_counter() - start) * 1000),
        "error": error,
    }


async def check_models() -> list[dict[str, Any]]:
    """Probe each provider's default model concurrently.

    :return: One ``check_model`` result per provider, in config order.
    """
    import asyncio

    pairs = [
        o for o in chat_model_options() if o["model"] == default_model(o["service"])
    ]
    return list(
        await asyncio.gather(*(check_model(o["service"], o["model"]) for o in pairs))
    )


def build_embedding_model(service: str, model: str):
    """Return a Pydantic AI embedding model for the requested service.

    Mirrors ``build_model``: provider packages are imported lazily, and each
    service gets the same credentials or base URL as its chat model.

    :param service: One of ``openai``, ``ollama``, ``bedrock``.
    :param model: Provider embedding model name.
    :return: A Pydantic AI ``EmbeddingModel`` instance.
    :raises NotImplementedError: For services without embedding support.
    :raises ValueError: If the service is unknown or its credentials are missing.
    """
    Model: Any
    Provider: Any
    kwargs: dict[str, Any]
    if service == "openai":
        from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel as Model
        from pydantic_ai.providers.openai import OpenAIProvider as Provider

        kwargs = {"api_key": require_env("OPENAI_API_KEY")}
    elif service == "ollama":
        # Ollama serves an OpenAI-compatible embeddings endpoint.
        from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel as Model
        from pydantic_ai.providers.ollama import OllamaProvider as Provider

        kwargs = {"base_url": os.getenv("OLLAMA_BASE_URL", OLLAMA_DEFAULT_BASE_URL)}
    elif service == "bedrock":
        from pydantic_ai.embeddings.bedrock import BedrockEmbeddingModel as Model
        from pydantic_ai.providers.bedrock import BedrockProvider as Provider

        kwargs = get_aws_config()
    elif service in ("anthropic", "groq"):
        raise NotImplementedError(
            f"{service} does not support text embeddings. Use openai, ollama, or bedrock."
        )
    else:
        raise ValueError(f"Unknown embedding service: {service}")
    return Model(model, provider=Provider(**kwargs))


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
    cache_file = home / ".aws" / "sso" / "cache" / (
        hashlib.sha1(cache_key.encode("utf-8"), usedforsecurity=False).hexdigest() + ".json"
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


# Selected provider for the app, overridable with the CHAT_SERVICE environment
# variable. Bedrock is the default so deploys work without editing code.
CHAT_SERVICE = os.getenv("CHAT_SERVICE", "bedrock")
