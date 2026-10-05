"""FastAPI server for the agent.

Routes: ``/`` serves the chat UI (``index.html``), ``/config`` reports the
selectable chat models and their status, ``POST /agent/activate``
builds and checks a model as soon as the UI picks it, and ``POST /agent/chat``
is the chat endpoint. Chat speaks Pydantic AI natively: request and response
bodies are ``ModelMessage`` JSON arrays. There is no flat-dict translation
layer.
"""

import asyncio
import logging
import os
import time
import webbrowser
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelResponse,
    sanitize_messages,
)

from agentseed.logger import setup_logging
from agentseed.providers import (
    ModelStatus,
    ModelRef,
    UnknownModelError,
    build_model,
    chat_model_options,
    check_model,
    check_models,
    first_line,
    resolve_chat_model,
)

load_dotenv()

# Pydantic AI prints a startup banner unless observability is configured.
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

# Single-file chat UI served at /.
INDEX_HTML = Path(__file__).parent / "index.html"

# Server-owned instructions. The client never supplies a system prompt.
INSTRUCTIONS = "You are a helpful assistant. Answer concisely."

# Added to INSTRUCTIONS for the first run after the conversation switches model.
SWITCH_NOTE = (
    "Earlier assistant turns in this conversation were written by {previous}. "
    "You are {current} and are taking over from here. Treat those turns as "
    "context, not as your own statements; correct them if they are wrong."
)

# ModelResponse.metadata key recording which "service:model" wrote a reply.
MODEL_LABEL_KEY = "agentseed_model"
# ModelResponse.metadata key holding the reply's price in USD (null if unknown).
COST_KEY = "agentseed_cost_usd"

logger = logging.getLogger(__name__)


def env_number(name: str, default: float, cast: type = float) -> Any:
    """Read a numeric setting from the environment, falling back on bad input.

    :param name: Environment variable name.
    :param default: Value used when the variable is unset or not a number.
    :param cast: ``float`` or ``int``.
    :return: The parsed value.
    """
    raw = os.getenv(name)
    if raw is None:
        return cast(default)
    try:
        return cast(float(raw))
    except (ValueError, OverflowError):
        logger.warning(f"Ignoring {name}={raw!r}: not a number, using {default:g}")
        return cast(default)


# Seconds a successful /agent/activate check is reused before re-probing.
ACTIVATE_TTL_S = env_number("ACTIVATE_TTL_S", 300)

# Maximum number of non-startup agents kept built at once.
AGENT_CACHE_SIZE = env_number("AGENT_CACHE_SIZE", 8, cast=int)


def build_agent(chat_model: ModelRef) -> Agent:
    """Build the app agent with its model and server-owned instructions.

    :param chat_model: Chat model, already resolved by ``resolve_chat_model``.
    :return: Configured ``Agent``.
    """
    logger.info(f"Initializing agent on '{chat_model}'")
    return Agent(build_model(*chat_model), instructions=INSTRUCTIONS)


@dataclass
class ResolvedAgent:
    """The agent a request resolved to, with the model it was built for."""

    agent: Agent
    chat_model: ModelRef


def requested_chat_model(service: str | None, model: str | None) -> ModelRef:
    """Resolve query params to a configured chat model, or reject the request.

    :param service: Optional provider (query param).
    :param model: Optional model (query param).
    :return: The ``ModelRef`` from ``resolve_chat_model``.
    :raises HTTPException: 400 for a chat_model not listed in models.json.
    """
    try:
        return resolve_chat_model(service, model)
    except UnknownModelError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def record_status(state: Any, status: ModelStatus) -> ModelStatus:
    """Save the latest status for a model on ``state.model_status``.

    A missing ``latency_ms`` keeps the previously recorded latency.

    :return: The saved status record.
    """
    previous = state.model_status.get(status.chat_model)
    if status.latency_ms is None and previous is not None:
        status = status.model_copy(update={"latency_ms": previous.latency_ms})
    state.model_status[status.chat_model] = status
    return status


def record_failure(state: Any, chat_model: ModelRef, e: BaseException) -> ModelStatus:
    """Record a model as not live because of ``e``."""
    return record_status(
        state, ModelStatus.of(chat_model, live=False, error=first_line(e))
    )


def previous_model(history: Sequence[ModelMessage]) -> str | None:
    """Return the label of whichever model wrote the last reply.

    Uses the ``agentseed_model`` metadata label, falling back to the reply's own
    ``provider_name:model_name`` for history saved before labelling existed.

    :param history: Message history.
    :return: ``service:model`` label, or None if no reply names its model.
    """
    for message in reversed(history):
        if not isinstance(message, ModelResponse):
            continue
        label = (message.metadata or {}).get(MODEL_LABEL_KEY)
        if label:
            return str(label)
        if message.model_name:
            if message.provider_name:
                return f"{message.provider_name}:{message.model_name}"
            return message.model_name
        return None
    return None


def response_cost(message: ModelResponse) -> float | None:
    """Price one reply in USD from its token usage, via genai-prices.

    :param message: A model reply carrying ``usage``, ``model_name`` and ``provider_name``.
    :return: Total price in USD, or None when the model has no known pricing (e.g. Ollama).
    """
    try:
        return float(message.cost().total_price)
    except Exception:  # LookupError for unpriced models, anything else is not fatal
        return None


def label_responses(messages: Sequence[ModelMessage], label: str) -> None:
    """Record which model wrote each reply, and what it cost, in ``ModelResponse.metadata``."""
    for message in messages:
        if isinstance(message, ModelResponse):
            message.metadata = {
                **(message.metadata or {}),
                MODEL_LABEL_KEY: label,
                COST_KEY: response_cost(message),
            }


def init_state(app: FastAPI) -> None:
    """Set up the ``app.state`` fields used by the routes.

    :param app: FastAPI application instance.
    """
    # The startup agent's model: the first chat model in models.json.
    app.state.default_chat_model = resolve_chat_model()
    # Everything below is keyed by ModelRef.
    # Built agents. The startup agent lives here under default_chat_model and is
    # never evicted; the rest form an LRU cache (see ensure_agent).
    app.state.agents = {}
    # Latest ModelStatus per model, from probes, activations and chats.
    app.state.model_status = {}
    # In-flight /agent/activate probes, so concurrent calls share one.
    app.state.activation_tasks = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the agent on startup and expose it on ``app.state``.

    A failed build is recorded rather than raised so the server still starts and
    reports the reason through /config.

    :param app: FastAPI application instance.
    :yields: None.
    """
    setup_logging()
    init_state(app)
    chat_model = app.state.default_chat_model
    try:
        app.state.agents[chat_model] = await asyncio.to_thread(build_agent, chat_model)
        logger.info(f"Agent initialised on '{chat_model}'")
    except Exception as e:
        record_failure(app.state, chat_model, e)
        logger.warning(f"Agent unavailable: {e}")
    # Probe providers in the background so a slow or dead provider never
    # delays startup; /config reports live=null for each until this finishes.
    probe = asyncio.create_task(probe_providers(app))
    yield
    probe.cancel()
    for task in list(app.state.activation_tasks.values()):
        task.cancel()


async def probe_providers(app: FastAPI) -> None:
    """Probe each provider's default model and record results on ``app.state``.

    :param app: FastAPI application whose ``model_status`` is updated.
    """
    try:
        results = await check_models()
    except Exception as e:
        logger.warning(f"Provider probe failed: {e}")
        return
    for status in results:
        record_status(app.state, status)
    live = [status.service for status in results if status.live]
    logger.info(f"Live providers: {', '.join(live) or 'none'}")


async def ensure_agent(state: Any, chat_model: ModelRef) -> Agent:
    """Return the built agent for a known model, building it off the event loop.

    All agents live in ``state.agents``. The startup model is rebuilt here if
    startup failed and is never evicted; other models are an LRU cache capped at
    ``AGENT_CACHE_SIZE``.

    :raises Exception: Whatever ``build_agent`` raises.
    """
    # Plain dicts keep insertion order: re-inserting marks a key most recently
    # used, so the earliest keys are the least recently used.
    agent = state.agents.pop(chat_model, None)
    if agent is None:
        agent = await asyncio.to_thread(build_agent, chat_model)
    state.agents[chat_model] = agent
    others = [k for k in state.agents if k != state.default_chat_model]
    for evicted in others[: max(0, len(others) - max(1, AGENT_CACHE_SIZE))]:
        del state.agents[evicted]
        logger.info(f"Evicted agent '{evicted}'")
    return agent


async def activate_model(state: Any, chat_model: ModelRef) -> ModelStatus:
    """Build the agent for a model, probe it, and record its status.

    :return: The status record.
    """
    try:
        await ensure_agent(state, chat_model)
    except Exception as e:
        state.agents.pop(chat_model, None)
        logger.warning(f"Agent '{chat_model}' unavailable: {e}")
        return record_failure(state, chat_model, e)
    status = await check_model(chat_model)
    if not status.live and chat_model != state.default_chat_model:
        state.agents.pop(chat_model, None)  # never reuse an agent that failed its check
    return record_status(state, status)


async def resolve_agent(
    request: Request, service: str | None = None, model: str | None = None
) -> ResolvedAgent:
    """Resolve query params to a built agent, or fail the request.

    The HTTP wrapper around ``ensure_agent``: missing params are filled in by
    ``resolve_chat_model`` and the agent is built if it isn't cached yet
    (including a startup agent whose first build failed).

    :param request: Incoming request, used to reach ``app.state``.
    :param service: Optional provider override (query param).
    :param model: Optional model override (query param); defaults to the
        provider's first configured model.
    :return: The agent and the model it serves.
    :raises HTTPException: 400 for an unknown model, 503 when the agent cannot
        be built.
    """
    chat_model = requested_chat_model(service, model)
    try:
        agent = await ensure_agent(request.app.state, chat_model)
    except Exception as e:
        logger.warning(f"Agent '{chat_model}' unavailable: {e}")
        raise HTTPException(status_code=503, detail=str(e)) from e
    return ResolvedAgent(agent, chat_model)


# Injected into routes that need the agent.
RESOLVED_AGENT = Annotated[ResolvedAgent, Depends(resolve_agent)]


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    :return: Configured FastAPI app
    """
    app = FastAPI(
        title="agentseed",
        description="Minimal agent server backed by Pydantic AI",
        version="0.1.0",
        lifespan=lifespan,
    )
    init_state(app)

    @app.post("/agent/activate")
    async def agent_activate(
        request: Request,
        service: str,
        model: str | None = None,
        force: bool = False,
    ) -> ModelStatus:
        """Build and probe a chat model as soon as the UI picks it.

        A model that does not work is still a successful check: the response is
        200 with ``live: false`` and ``error`` set. A live result is reused for
        ``ACTIVATE_TTL_S`` seconds unless ``force`` is set; concurrent calls for
        the same model share one probe.

        :param request: Incoming request, used to reach ``app.state``.
        :param service: Provider name.
        :param model: Model name; defaults to the provider's first model.
        :param force: Re-probe even if a fresh live status exists.
        :return: Status record ``{service, model, live, latency_ms, error, checked_at}``.
        """
        state = request.app.state
        chat_model = requested_chat_model(service, model)

        status = state.model_status.get(chat_model)
        if not force and status and status.is_fresh(ACTIVATE_TTL_S):
            try:
                await ensure_agent(state, chat_model)  # rebuild if evicted
                return status
            except Exception as e:
                return record_failure(state, chat_model, e)

        task = state.activation_tasks.get(chat_model)
        if task is None:
            task = asyncio.create_task(activate_model(state, chat_model))
            state.activation_tasks[chat_model] = task

            def forget(done: asyncio.Task) -> None:
                if state.activation_tasks.get(chat_model) is done:
                    del state.activation_tasks[chat_model]

            task.add_done_callback(forget)
        # Shield so one client disconnecting doesn't cancel a shared probe.
        return await asyncio.shield(task)

    @app.post("/agent/chat")
    async def agent_chat(request: Request, resolved: RESOLVED_AGENT) -> Response:
        """Chat using native Pydantic AI message history.

        Request and response bodies are ``ModelMessage`` JSON arrays (the
        ``ModelMessagesTypeAdapter`` format). The last message is the new user
        turn; earlier messages are history. Optional ``service`` and
        ``model`` query params select a chat model from models.json.

        New replies are labelled with ``metadata.agentseed_model``. When the
        last reply came from a different model, ``SWITCH_NOTE`` is added to the
        instructions for this run.

        :param request: Incoming request carrying the raw message history.
        :param resolved: Injected agent and the model it serves.
        :return: The full updated message history as JSON.
        """
        try:
            history = ModelMessagesTypeAdapter.validate_json(await request.body())
        except ValidationError as e:
            raise HTTPException(
                status_code=400, detail=f"invalid message history: {e}"
            ) from e
        if not history:
            raise HTTPException(status_code=400, detail="message history is empty")
        # The server owns the instructions, so drop any client-supplied prompt.
        history = sanitize_messages(history)

        state = request.app.state
        chat_model = resolved.chat_model
        current = str(chat_model)
        previous = previous_model(history)
        instructions = (
            SWITCH_NOTE.format(previous=previous, current=current)
            if previous and previous != current
            else None
        )
        if instructions:
            logger.info(f"Model switched from '{previous}' to '{current}'")
        try:
            result = await resolved.agent.run(
                message_history=history, instructions=instructions
            )
        except Exception as e:
            logger.error(f"Agent error: {e}")
            record_failure(state, chat_model, e)
            raise HTTPException(status_code=500, detail=str(e)) from e
        record_status(state, ModelStatus.of(chat_model, live=True))

        messages = result.all_messages()
        label_responses(messages[len(history) :], current)
        return Response(
            ModelMessagesTypeAdapter.dump_json(messages),
            media_type="application/json",
        )

    @app.get("/config")
    async def config(request: Request) -> dict[str, Any]:
        """Return agent configuration, availability, and any initialization error.

        :param request: Incoming request, used to reach ``app.state``.
        :return: Config dict. Example::

            {
                "service": "bedrock",
                "available": true,
                "error": null,
                "model": "amazon.nova-pro-v1:0",
                "models": [
                    {
                        "service": "bedrock",
                        "model": "amazon.nova-pro-v1:0",
                        "live": true,
                        "error": null,
                        "latency_ms": 412,
                        "checked": true
                    }
                ]
            }

        ``live`` is null while the startup probe for that provider is pending.
        ``checked`` is false when the status is borrowed from the provider's
        default model rather than measured for this exact model.
        """
        state = request.app.state
        default = state.default_chat_model
        models = []
        for chat_model in chat_model_options():
            status = state.model_status.get(chat_model)
            checked = status is not None
            if status is None:
                # Borrow the provider's default model status until this one is checked.
                status = state.model_status.get(resolve_chat_model(chat_model.service))
            models.append(
                {
                    **chat_model._asdict(),
                    "live": status.live if status else None,
                    "error": status.error if status else None,
                    "latency_ms": status.latency_ms if checked else None,
                    "checked": checked,
                }
            )
        default_status = state.model_status.get(default)
        available = default in state.agents
        return {
            **default._asdict(),
            "available": available,
            "error": None if available or not default_status else default_status.error,
            "models": models,
        }

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        """Serve the single-file chat UI (``agentseed/index.html``)."""
        return FileResponse(INDEX_HTML, media_type="text/html")

    return app


def wait_and_open_browser(check_url: str, open_url: str) -> None:
    """Poll check_url until ready, then open open_url in the browser.

    :param check_url: URL to poll for readiness (expects HTTP 200).
    :param open_url: URL to open in the browser once ready.
    :return: None.
    """
    max_retries = 60
    retry_count = 0

    while retry_count < max_retries:
        try:
            # verify=False so the local readiness check also works with the
            # self-signed certificates produced by --ssl.
            response = httpx.get(check_url, timeout=1, verify=False)  # noqa: S501
            if response.status_code == 200:
                webbrowser.open(open_url)
                logger.info(f"Opening {open_url} in browser...")
                return
        except httpx.HTTPError:
            pass  # server not accepting connections yet; retry

        time.sleep(0.5)
        retry_count += 1

    try:
        webbrowser.open(open_url)
        logger.info(f"Opening {open_url} in browser (timeout waiting for ready)...")
    except Exception as e:
        logger.error(f"Could not open browser: {e}")


# module-level app instance used by uvicorn reload mode
app = create_app()
