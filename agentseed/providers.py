"""Provider plumbing for the agent.

Builds a Pydantic AI model or embedding model for each supported service and
delegates AWS and Bedrock setup to ``agentseed.aws``.
"""

import json
import logging
import os
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

from pydantic import BaseModel, Field
from pydantic_ai.embeddings import infer_embedding_model
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import infer_model
from pydantic_ai.providers import infer_provider

logger = logging.getLogger(__name__)

_SERVICES = ("openai", "anthropic", "groq", "ollama", "bedrock")
_UNSUPPORTED_EMBEDDING_SERVICES = ("anthropic", "groq")


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
def chat_model_options() -> tuple[ModelRef, ...]:
    """Return every selectable chat model from models.json.

    :return: ``ModelRef`` per model, in config order.
    :raises ValueError: If models.json cannot be read or parsed.
    """
    config_path = Path(__file__).parent / "models.json"
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"Could not load models config '{config_path}': {e}") from e
    logger.info(f"Loaded selectable models from '{config_path}'")
    return tuple(
        ModelRef(service, model)
        for service, models in config.get("chat_models", {}).items()
        for model in (models if isinstance(models, list) else [models])
    )


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


def build_provider(service: str) -> Any:
    """Return a Pydantic AI provider with this service's credentials or base URL.

    Shared by ``build_model`` and ``build_embedding_model``. Pydantic AI reads
    the standard env vars itself (``OPENAI_API_KEY`` and so on). Only Ollama
    (default base URL) and Bedrock (AWS checks, logging) need custom setup.

    :param service: One of ``openai``, ``anthropic``, ``groq``, ``ollama``, ``bedrock``.
    :return: A configured provider instance.
    :raises ValueError: If the service is unknown or its credentials are missing.
    """
    if service == "ollama":
        from pydantic_ai.providers.ollama import OllamaProvider
        OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/v1"

        return OllamaProvider(
            base_url=os.getenv("OLLAMA_BASE_URL", OLLAMA_DEFAULT_BASE_URL)
        )
    if service == "bedrock":
        from agentseed.aws import build_bedrock_provider

        return build_bedrock_provider()
    if service not in _SERVICES:
        raise ValueError(f"Unknown provider: {service}")
    try:
        return infer_provider(service)
    except UserError as e:
        raise ValueError(str(e)) from e


def build_model(service: str, model: str):
    """Return a Pydantic AI chat model for the requested service.

    :param service: One of ``openai``, ``anthropic``, ``groq``, ``ollama``, ``bedrock``.
    :param model: Provider model name.
    :return: A Pydantic AI ``Model`` instance.
    :raises ValueError: If the service is unknown or its credentials are missing.
    """
    if service not in _SERVICES:
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
    if service in _UNSUPPORTED_EMBEDDING_SERVICES:
        raise NotImplementedError(
            f"{service} does not support text embeddings. Use openai, ollama, or bedrock."
        )
    if service not in _SERVICES:
        raise ValueError(f"Unknown embedding service: {service}")
    return infer_embedding_model(f"{service}:{model}", provider_factory=build_provider)


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
