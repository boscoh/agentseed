"""FastAPI server for the agent.

Routes: ``/`` serves the chat UI (``index.html``), ``/config`` reports the
selectable provider/model pairs and their status, ``POST /agent/activate``
builds and checks a pair as soon as the UI picks it, and ``POST /agent/chat``
is the chat endpoint. Chat speaks Pydantic AI natively: request and response
bodies are ``ModelMessage`` JSON arrays. There is no flat-dict translation
layer.
"""

import asyncio
import logging
import os
import time
import webbrowser
from collections import OrderedDict
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
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
    UnknownModelError,
    build_model,
    chat_model_options,
    check_model,
    check_models,
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

Key = tuple[str, str]


def build_agent(service: str, model: str) -> Agent:
    """Build the app agent with its model and server-owned instructions.

    :param service: Provider name, already resolved by ``resolve_chat_model``.
    :param model: Model name, already resolved by ``resolve_chat_model``.
    :return: Configured ``Agent``.
    """
    logger.info(f"Initializing agent on '{model_label(service, model)}'")
    return Agent(build_model(service, model), instructions=INSTRUCTIONS)


class AgentCache:
    """Least-recently-used cache of built agents keyed by ``(service, model)``."""

    def __init__(self, maxsize: int = AGENT_CACHE_SIZE) -> None:
        """Create an empty cache.

        :param maxsize: Maximum number of agents kept; at least 1.
        """
        self.maxsize = max(1, maxsize)
        self._agents: OrderedDict[Key, Agent] = OrderedDict()

    def get(self, key: Key) -> Agent | None:
        """Return the cached agent and mark it most recently used.

        :param key: ``(service, model)`` pair.
        :return: The agent, or None if not cached.
        """
        agent = self._agents.get(key)
        if agent is not None:
            self._agents.move_to_end(key)
        return agent

    def put(self, key: Key, agent: Agent) -> None:
        """Store an agent, dropping the least recently used one when full.

        :param key: ``(service, model)`` pair.
        :param agent: Agent to cache.
        """
        self._agents[key] = agent
        self._agents.move_to_end(key)
        while len(self._agents) > self.maxsize:
            evicted, _ = self._agents.popitem(last=False)
            logger.info(f"Evicted agent '{model_label(*evicted)}'")

    def pop(self, key: Key) -> None:
        """Remove an agent if present.

        :param key: ``(service, model)`` pair.
        """
        self._agents.pop(key, None)

    def __contains__(self, key: object) -> bool:
        return key in self._agents

    def __len__(self) -> int:
        return len(self._agents)

    def keys(self) -> list[Key]:
        """Return cached keys, least recently used first."""
        return list(self._agents)


@dataclass
class Selection:
    """The agent chosen for a request, with the pair it was built for."""

    agent: Agent
    service: str
    model: str


def model_label(service: str, model: str) -> str:
    """Return the ``service:model`` label used in metadata and the switch note."""
    return f"{service}:{model}"


def requested_pair(service: str | None, model: str | None) -> Key:
    """Resolve query params to a configured pair, or reject the request.

    :param service: Optional provider (query param).
    :param model: Optional model (query param).
    :return: ``(service, model)`` from ``resolve_chat_model``.
    :raises HTTPException: 400 for a pair not listed in models.json.
    """
    try:
        return resolve_chat_model(service, model)
    except UnknownModelError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def first_line(e: BaseException) -> str:
    """Return a one-line error message for status records."""
    text = str(e).strip()
    return text.splitlines()[0] if text else type(e).__name__


def record_status(
    state: Any,
    service: str,
    model: str,
    live: bool,
    error: str | None,
    latency_ms: int | None = None,
) -> dict[str, Any]:
    """Save the latest status for a pair on ``state.model_status``.

    A missing ``latency_ms`` keeps the previously recorded latency.

    :return: The saved status record.
    """
    previous = state.model_status.get((service, model), {})
    status = {
        "service": service,
        "model": model,
        "live": live,
        "latency_ms": latency_ms if latency_ms is not None else previous.get("latency_ms"),
        "error": error,
        "checked_at": datetime.now(UTC).isoformat(),
    }
    state.model_status[(service, model)] = status
    return status


def is_fresh(status: dict[str, Any] | None) -> bool:
    """Return True for a live status checked within ``ACTIVATE_TTL_S``."""
    if not status or not status.get("live"):
        return False
    checked_at = datetime.fromisoformat(status["checked_at"])
    age = (datetime.now(UTC) - checked_at).total_seconds()
    return age < ACTIVATE_TTL_S


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
                return model_label(message.provider_name, message.model_name)
            return message.model_name
        return None
    return None


def label_responses(messages: Sequence[ModelMessage], label: str) -> None:
    """Record which model wrote each reply in ``ModelResponse.metadata``."""
    for message in messages:
        if isinstance(message, ModelResponse):
            message.metadata = {**(message.metadata or {}), MODEL_LABEL_KEY: label}


def init_state(app: FastAPI) -> None:
    """Set up the ``app.state`` fields used by the routes.

    :param app: FastAPI application instance.
    """
    app.state.agent = None
    app.state.agent_error = None
    # The startup agent's pair: the first chat model in models.json.
    app.state.default_pair = resolve_chat_model()
    app.state.agents = AgentCache(AGENT_CACHE_SIZE)
    app.state.model_status = {}
    app.state.activation_tasks = {}
    app.state.provider_status = {}


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
    try:
        app.state.agent = await asyncio.to_thread(build_agent, *app.state.default_pair)
        logger.info(f"Agent initialised on '{model_label(*app.state.default_pair)}'")
    except Exception as e:
        app.state.agent_error = str(e)
        logger.warning(f"Agent unavailable: {app.state.agent_error}")
    # Probe providers in the background so a slow or dead provider never
    # delays startup; /config reports live=null for each until this finishes.
    probe = asyncio.create_task(probe_providers(app))
    yield
    probe.cancel()
    for task in list(app.state.activation_tasks.values()):
        task.cancel()


async def probe_providers(app: FastAPI) -> None:
    """Probe each provider's default model and record results on ``app.state``.

    :param app: FastAPI application whose ``provider_status`` and
        ``model_status`` are updated.
    """
    try:
        results = await check_models()
    except Exception as e:
        logger.warning(f"Provider probe failed: {e}")
        return
    app.state.provider_status = {r["service"]: r for r in results}
    for r in results:
        record_status(
            app.state, r["service"], r["model"], r["live"], r["error"], r["latency_ms"]
        )
    live = [r["service"] for r in results if r["live"]]
    logger.info(f"Live providers: {', '.join(live) or 'none'}")


async def ensure_agent(state: Any, service: str, model: str) -> Agent:
    """Return the built agent for a known pair, building it off the event loop.

    The startup pair uses ``state.agent`` (rebuilt if startup failed); other
    pairs go through the LRU ``state.agents`` cache.

    :raises Exception: Whatever ``build_agent`` raises.
    """
    key = (service, model)
    if key == state.default_pair:
        if state.agent is None:
            state.agent = await asyncio.to_thread(build_agent, service, model)
            state.agent_error = None
        return state.agent
    agent = state.agents.get(key)
    if agent is None:
        agent = await asyncio.to_thread(build_agent, service, model)
        state.agents.put(key, agent)
    return agent


async def activate_pair(state: Any, service: str, model: str) -> dict[str, Any]:
    """Build the agent for a pair, probe the model, and record its status.

    :return: The status record.
    """
    key = (service, model)
    try:
        await ensure_agent(state, service, model)
    except Exception as e:
        state.agents.pop(key)
        logger.warning(f"Agent '{model_label(service, model)}' unavailable: {e}")
        return record_status(state, service, model, False, first_line(e))
    result = await check_model(service, model)
    if not result["live"]:
        state.agents.pop(key)  # never reuse an agent that failed its check
    return record_status(
        state, service, model, result["live"], result["error"], result["latency_ms"]
    )


async def require_agent(
    request: Request, service: str | None = None, model: str | None = None
) -> Selection:
    """Return the agent for the requested provider/model, or fail the request.

    Missing params are filled in by ``resolve_chat_model``. The default pair
    uses the startup agent; other pairs come from the LRU cache, built on first
    use (or by ``/agent/activate``).

    :param request: Incoming request, used to reach ``app.state``.
    :param service: Optional provider override (query param).
    :param model: Optional model override (query param); defaults to the
        provider's first configured model.
    :return: The agent and the pair it serves.
    :raises HTTPException: 400 for an unknown pair, 503 when the agent cannot
        be initialised.
    """
    state = request.app.state
    service, model = requested_pair(service, model)
    if (service, model) == state.default_pair and state.agent is None:
        # Startup failed; don't retry on every chat (/agent/activate retries).
        raise HTTPException(
            status_code=503, detail=state.agent_error or "Agent not initialized"
        )
    try:
        agent = await ensure_agent(state, service, model)
    except Exception as e:
        logger.warning(f"Agent '{model_label(service, model)}' unavailable: {e}")
        raise HTTPException(status_code=503, detail=str(e)) from e
    return Selection(agent, service, model)


# Injected into routes that need the agent.
SELECTED = Annotated[Selection, Depends(require_agent)]


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
    ) -> dict[str, Any]:
        """Build and probe a provider/model pair as soon as the UI picks it.

        A model that does not work is still a successful check: the response is
        200 with ``live: false`` and ``error`` set. A live result is reused for
        ``ACTIVATE_TTL_S`` seconds unless ``force`` is set; concurrent calls for
        the same pair share one probe.

        :param request: Incoming request, used to reach ``app.state``.
        :param service: Provider name.
        :param model: Model name; defaults to the provider's first model.
        :param force: Re-probe even if a fresh live status exists.
        :return: Status record ``{service, model, live, latency_ms, error, checked_at}``.
        """
        state = request.app.state
        service, model = requested_pair(service, model)
        key = (service, model)

        status = state.model_status.get(key)
        if not force and is_fresh(status):
            try:
                await ensure_agent(state, service, model)  # rebuild if evicted
                return status
            except Exception as e:
                return record_status(state, service, model, False, first_line(e))

        task = state.activation_tasks.get(key)
        if task is None:
            task = asyncio.create_task(activate_pair(state, service, model))
            state.activation_tasks[key] = task

            def forget(done: asyncio.Task) -> None:
                if state.activation_tasks.get(key) is done:
                    del state.activation_tasks[key]

            task.add_done_callback(forget)
        # Shield so one client disconnecting doesn't cancel a shared probe.
        return await asyncio.shield(task)

    @app.post("/agent/chat")
    async def agent_chat(request: Request, selection: SELECTED) -> Response:
        """Chat using native Pydantic AI message history.

        Request and response bodies are ``ModelMessage`` JSON arrays (the
        ``ModelMessagesTypeAdapter`` format). The last message is the new user
        turn; earlier messages are history. Optional ``service`` and
        ``model`` query params select a provider/model pair from models.json.

        New replies are labelled with ``metadata.agentseed_model``. When the
        last reply came from a different model, ``SWITCH_NOTE`` is added to the
        instructions for this run.

        :param request: Incoming request carrying the raw message history.
        :param selection: Injected agent and the pair it serves.
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
        service, model = selection.service, selection.model
        current = model_label(service, model)
        previous = previous_model(history)
        instructions = (
            SWITCH_NOTE.format(previous=previous, current=current)
            if previous and previous != current
            else None
        )
        if instructions:
            logger.info(f"Model switched from '{previous}' to '{current}'")
        try:
            result = await selection.agent.run(
                message_history=history, instructions=instructions
            )
        except Exception as e:
            logger.error(f"Agent error: {e}")
            record_status(state, service, model, False, first_line(e))
            raise HTTPException(status_code=500, detail=str(e)) from e
        record_status(state, service, model, True, None)

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
        models = []
        for option in chat_model_options():
            status = state.model_status.get((option["service"], option["model"]))
            checked = status is not None
            if status is None:
                status = state.provider_status.get(option["service"])
            models.append(
                {
                    **option,
                    "live": status["live"] if status else None,
                    "error": status["error"] if status else None,
                    "latency_ms": status.get("latency_ms") if checked else None,
                    "checked": checked,
                }
            )
        return {
            "service": state.default_pair[0],
            "available": state.agent is not None,
            "error": state.agent_error,
            "model": state.default_pair[1],
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
