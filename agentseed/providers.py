"""Provider plumbing for the agent.

Builds a Pydantic AI chat model for each provider listed in ``models.json`` and
delegates AWS and Bedrock setup to ``agentseed.aws``.
"""

import logging
import os
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import infer_model
from pydantic_ai.providers import infer_provider

logger = logging.getLogger(__name__)


class ProviderSpec(BaseModel):
    """One provider entry from models.json; the provider name is the dict key.

    An unknown key or a wrong type raises ``ValidationError`` at load, so a bad
    hand edit in models.json cannot pass in silence.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    # one env var set means configured; empty means the provider needs no credentials
    env: tuple[str, ...] = ()
    base_url: str | None = None
    # chat models; the first model is the default and probe model
    models: tuple[str, ...] = Field(alias="chat")


class ModelsConfig(BaseModel):
    """The models.json document: provider name to provider entry."""

    model_config = ConfigDict(extra="forbid")

    providers: dict[str, ProviderSpec]


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


@lru_cache
def providers() -> dict[str, ProviderSpec]:
    """Return every provider listed in models.json, in file order.

    :return: ``ProviderSpec`` by provider name.
    :raises ValueError: If models.json cannot be read or is malformed.
    """
    config_path = Path(__file__).parent / "models.json"
    try:
        config = ModelsConfig.model_validate_json(config_path.read_text())
    except (OSError, ValidationError) as e:
        raise ValueError(f"Could not load models config '{config_path}': {e}") from e
    specs = dict(config.providers)
    empty = [name for name, spec in specs.items() if not spec.models]
    if empty:
        raise ValueError(f"Providers with no chat models in '{config_path}': {empty}")
    logger.info(f"Loaded {len(specs)} providers from '{config_path}'")
    return specs


def chat_model_options() -> tuple[ModelRef, ...]:
    """Return every selectable chat model from models.json.

    :return: ``ModelRef`` per model, in file order.
    """
    return tuple(
        ModelRef(name, model)
        for name, spec in providers().items()
        for model in spec.models
    )


def is_configured(service: str) -> bool:
    """Return True if the provider's credentials look present.

    A provider with no ``env`` entry (Ollama) needs no credentials. Bedrock
    also counts AWS config files, because boto3 reads them without env vars.

    :param service: Provider name from models.json.
    """
    spec = providers()[service]
    if not spec.env or any(os.environ.get(var) for var in spec.env):
        return True
    if service == "bedrock":
        from agentseed.aws import has_aws_files

        return has_aws_files()
    return False


def not_configured_message(service: str) -> str:
    """Return the error text for a provider with no credentials."""
    return f"{service} is not configured: set {' or '.join(providers()[service].env)}"


def resolve_chat_model(
    service: str | None = None, model: str | None = None
) -> ModelRef:
    """Fill in defaults for a provider/model and check it is listed.

    This is the one place chat defaults are decided:

    - no service: the first configured provider in models.json, else the first
      provider (so the failure shows in /config);
    - no model: that provider's first model.

    :param service: Provider name, or None for the default provider.
    :param model: Model name, or None for the provider's first model.
    :return: A ``ModelRef`` guaranteed to be listed in models.json.
    :raises UnknownModelError: If the provider or model is not listed.
    """
    specs = providers()
    if service is None:
        service = next((name for name in specs if is_configured(name)), None)
        service = service or next(iter(specs))
    spec = specs.get(service)
    if spec is None:
        raise UnknownModelError(f"unknown provider: {service}")
    if model is None:
        model = spec.models[0]
    chat_model = ModelRef(service, model)
    if model not in spec.models:
        raise UnknownModelError(f"unknown provider/model: {chat_model}")
    return chat_model


def build_provider(service: str) -> Any:
    """Return a Pydantic AI provider with this service's credentials or base URL.

    Pydantic AI reads the standard env vars itself (``OPENAI_API_KEY`` and so
    on). Only Ollama (base URL) and Bedrock (AWS checks, logging) need custom
    setup.

    :param service: A provider name from models.json.
    :return: A configured provider instance.
    :raises ValueError: If the service is unknown or its credentials are missing.
    """
    spec = providers().get(service)
    if spec is None:
        raise ValueError(f"Unknown provider: {service}")
    if not is_configured(service):
        raise ValueError(not_configured_message(service))
    if service == "ollama":
        from pydantic_ai.providers.ollama import OllamaProvider

        return OllamaProvider(base_url=os.getenv("OLLAMA_BASE_URL", spec.base_url))
    if service == "bedrock":
        from agentseed.aws import build_bedrock_provider

        return build_bedrock_provider()
    try:
        return infer_provider(service)
    except UserError as e:
        raise ValueError(str(e)) from e


def build_model(service: str, model: str):
    """Return a Pydantic AI chat model for the requested service.

    :param service: A provider name from models.json.
    :param model: Provider model name.
    :return: A Pydantic AI ``Model`` instance.
    :raises ValueError: If the service is unknown or its credentials are missing.
    """
    if service not in providers():
        raise ValueError(f"Unknown chat client type: {service}")
    return infer_model(f"{service}:{model}", provider_factory=build_provider)


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
    """Check each provider's default model, probing configured ones concurrently.

    A provider without credentials is reported as not live, without a probe.

    :return: One status per provider, in file order.
    """
    import asyncio

    results: dict[str, ModelStatus] = {}
    probes = []
    for service in providers():
        chat_model = resolve_chat_model(service)
        if is_configured(service):
            probes.append(check_model(chat_model))
        else:
            results[service] = ModelStatus.of(
                chat_model, live=False, error=not_configured_message(service)
            )
    for status in await asyncio.gather(*probes):
        results[status.service] = status
    return [results[service] for service in providers()]
