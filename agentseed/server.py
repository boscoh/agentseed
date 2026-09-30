"""FastAPI server for the agent.

Routes: ``/`` serves the chat UI (``index.html``), ``/health`` is a
liveness check, ``/config`` reports the selected provider/model, and
``POST /agent/chat`` is the chat endpoint. Chat speaks Pydantic AI natively:
request and response bodies are ``ModelMessage`` JSON arrays. There is no
flat-dict translation layer.
"""

import asyncio
import logging
import os
import time
import webbrowser
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessagesTypeAdapter, sanitize_messages

from agentseed.logger import setup_logging
from agentseed.providers import (
    CHAT_SERVICE,
    build_model,
    chat_model_options,
    check_models,
    default_model,
)

load_dotenv()

# Pydantic AI prints a startup banner unless observability is configured.
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

# Single-file chat UI served at /.
INDEX_HTML = Path(__file__).parent / "index.html"

# Server-owned instructions. The client never supplies a system prompt.
INSTRUCTIONS = "You are a helpful assistant. Answer concisely."

logger = logging.getLogger(__name__)


def build_agent(service: str = CHAT_SERVICE, model: str | None = None) -> Agent:
    """Build the app agent with its model and server-owned instructions.

    :param service: One of ``bedrock``, ``openai``, ``anthropic``, ``groq``, ``ollama``.
    :param model: Provider model name; defaults to the first entry in models.json.
    :return: Configured ``Agent``.
    """
    model = model or default_model(service)
    if model is None:
        raise ValueError(f"No chat models configured for '{service}' in models.json")
    logger.info(f"Initializing agent on '{service}:{model}'")
    return Agent(build_model(service, model), instructions=INSTRUCTIONS)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the agent on startup and expose it on ``app.state``.

    A failed build is recorded rather than raised so the server still starts and
    reports the reason through /config.

    :param app: FastAPI application instance.
    :yields: None.
    """
    setup_logging()
    app.state.agent = None
    app.state.agent_error = None
    app.state.model_name = default_model(CHAT_SERVICE)
    try:
        app.state.agent = build_agent()
        logger.info(f"Agent initialised on '{CHAT_SERVICE}:{app.state.model_name}'")
    except Exception as e:
        app.state.agent_error = str(e)
        logger.warning(f"Agent unavailable: {app.state.agent_error}")
    # Probe providers in the background so a slow or dead provider never
    # delays startup; /config reports live=null for each until this finishes.
    app.state.provider_status = {}
    probe = asyncio.create_task(probe_providers(app))
    yield
    probe.cancel()


async def probe_providers(app: FastAPI) -> None:
    """Probe each provider's default model and record results on ``app.state``.

    :param app: FastAPI application whose ``provider_status`` is updated.
    """
    try:
        results = await check_models()
    except Exception as e:
        logger.warning(f"Provider probe failed: {e}")
        return
    record_provider_status(app, results)
    live = [r["service"] for r in results if r["live"]]
    logger.info(f"Live providers: {', '.join(live) or 'none'}")


def record_provider_status(app: FastAPI, results: list[dict[str, Any]]) -> None:
    """Store default-model probe results keyed by service.

    A provider's default model stands in for the provider, so non-default
    results (from ``all_models`` probes) are ignored here.

    :param app: FastAPI application whose ``provider_status`` is updated.
    :param results: ``check_model`` results.
    """
    statuses = getattr(app.state, "provider_status", None)
    if statuses is None:
        statuses = app.state.provider_status = {}
    for r in results:
        if r["model"] == default_model(r["service"]):
            statuses[r["service"]] = r


def require_agent(
    request: Request, service: str | None = None, model: str | None = None
) -> Agent:
    """Return the agent for the requested provider/model, or fail the request.

    With no ``service``/``model`` query params (or the startup pair) the startup
    agent is used. Other pairs must be listed in models.json; their agents are
    built on first use and cached on ``app.state.agents``.

    :param request: Incoming request, used to reach ``app.state``.
    :param service: Optional provider override (query param).
    :param model: Optional model override (query param); defaults to the
        provider's first configured model.
    :return: The initialised agent.
    :raises HTTPException: 400 for an unknown pair, 503 when the agent cannot
        be initialised.
    """
    state = request.app.state
    service = service or CHAT_SERVICE
    model = model or default_model(service)
    if service == CHAT_SERVICE and model == getattr(state, "model_name", None):
        agent = getattr(state, "agent", None)
        if agent is None:
            detail = getattr(state, "agent_error", None)
            raise HTTPException(
                status_code=503, detail=detail or "Agent not initialized"
            )
        return agent

    if model is None or {"service": service, "model": model} not in chat_model_options():
        raise HTTPException(
            status_code=400, detail=f"unknown provider/model: {service}:{model}"
        )
    agents: dict[tuple[str, str], Agent] = getattr(state, "agents", None) or {}
    state.agents = agents
    key = (service, model)
    if key not in agents:
        try:
            agents[key] = build_agent(service, model)
        except Exception as e:
            logger.warning(f"Agent '{service}:{model}' unavailable: {e}")
            raise HTTPException(status_code=503, detail=str(e)) from e
    return agents[key]


# Injected into routes that need the agent.
AGENT = Annotated[Agent, Depends(require_agent)]


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

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Return a simple liveness response."""
        return {"service": "agentseed", "status": "ok"}

    @app.post("/agent/chat")
    async def agent_chat(request: Request, agent: AGENT) -> Response:
        """Chat using native Pydantic AI message history.

        Request and response bodies are ``ModelMessage`` JSON arrays (the
        ``ModelMessagesTypeAdapter`` format). The last message is the new user
        turn; earlier messages are history. Optional ``service`` and
        ``model`` query params select a provider/model pair from models.json.

        :param request: Incoming request carrying the raw message history.
        :param agent: Injected Pydantic AI agent.
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
        try:
            result = await agent.run(message_history=history)
        except Exception as e:
            logger.error(f"Agent error: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e
        return Response(result.all_messages_json(), media_type="application/json")

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
                        "error": null
                    }
                ]
            }

        ``live`` is null while the startup probe for that provider is pending.
        """
        statuses = getattr(request.app.state, "provider_status", {})
        models = []
        for option in chat_model_options():
            status = statuses.get(option["service"])
            models.append(
                {
                    **option,
                    "live": status["live"] if status else None,
                    "error": status["error"] if status else None,
                }
            )
        return {
            "service": CHAT_SERVICE,
            "available": getattr(request.app.state, "agent", None) is not None,
            "error": getattr(request.app.state, "agent_error", None),
            "model": getattr(request.app.state, "model_name", None),
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
