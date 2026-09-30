"""Tests for the native Pydantic AI /agent/chat endpoint (offline, no network)."""

import httpx
import pytest
from fastapi import FastAPI
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelRequest,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel

import agentseed.server as server


@pytest.fixture(autouse=True)
def no_real_agents(monkeypatch):
    """Fail loudly if a test would build a real (networked) agent."""

    def refuse(service, model=None):
        raise AssertionError(f"test tried to build a real agent: {service}:{model}")

    monkeypatch.setattr(server, "build_agent", refuse)


def _user_history(text: str) -> bytes:
    """Return a one-message ModelMessage history, as the client would POST it."""
    return ModelMessagesTypeAdapter.dump_json(
        [ModelRequest(parts=[UserPromptPart(content=text)])]
    )


def _test_app() -> FastAPI:
    """Return an app whose startup agent is an offline TestModel.

    httpx's ASGITransport doesn't run the lifespan, so set the state it would:
    the agent *and* its model name, otherwise ``require_agent`` treats a
    default request as a different pair and tries to build a real agent.
    """
    app = server.create_app()
    app.state.agent = Agent(TestModel(), instructions="test")
    app.state.model_name = server.default_model(server.CHAT_SERVICE)
    return app


async def test_agent_chat_roundtrip():
    app = _test_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
        res = await ac.post(
            "/agent/chat",
            content=_user_history("hello"),
            headers={"Content-Type": "application/json"},
        )
    assert res.status_code == 200
    history = ModelMessagesTypeAdapter.validate_json(res.content)
    assert history[-1].kind == "response"


async def test_agent_chat_rejects_empty_history():
    app = _test_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
        res = await ac.post("/agent/chat", content=b"[]")
    assert res.status_code == 400


async def test_agent_chat_rejects_invalid_json():
    app = _test_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
        res = await ac.post("/agent/chat", content=b"not json")
    assert res.status_code == 400


async def test_config_lists_model_options():
    app = _test_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
        res = await ac.get("/config")
    models = res.json()["models"]
    pairs = [{"service": m["service"], "model": m["model"]} for m in models]
    assert {"service": "openai", "model": "gpt-4o"} in pairs
    # No startup probe has run in tests, so liveness is unknown.
    assert all(m["live"] is None for m in models)


async def test_agent_chat_rejects_unknown_model():
    app = _test_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
        res = await ac.post(
            "/agent/chat?service=openai&model=nope",
            content=_user_history("hello"),
        )
    assert res.status_code == 400


async def test_agent_chat_uses_selected_model():
    app = _test_app()
    app.state.agents = {("openai", "gpt-4o"): Agent(TestModel(), instructions="t")}
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
        res = await ac.post(
            "/agent/chat?service=openai&model=gpt-4o",
            content=_user_history("hello"),
        )
    assert res.status_code == 200


def test_build_anthropic_model(monkeypatch):
    from pydantic_ai.models.anthropic import AnthropicModel

    from agentseed.providers import build_model

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    model = build_model("anthropic", "claude-sonnet-4-5")
    assert isinstance(model, AnthropicModel)
    assert model.model_name == "claude-sonnet-4-5"


def test_build_anthropic_model_requires_key(monkeypatch):
    from agentseed.providers import build_model

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        build_model("anthropic", "claude-sonnet-4-5")


async def test_config_marks_dead_providers():
    app = _test_app()
    app.state.provider_status = {
        "groq": {"live": False, "error": "GROQ_API_KEY environment variable is not set."},
        "anthropic": {"live": True, "error": None},
    }
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
        models = (await ac.get("/config")).json()["models"]
    by_service = {m["service"]: m for m in models}
    assert by_service["groq"]["live"] is False
    assert "GROQ_API_KEY" in by_service["groq"]["error"]
    assert by_service["anthropic"]["live"] is True
    assert by_service["openai"]["live"] is None  # not probed yet


async def test_index_served_at_root():
    app = _test_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
        res = await ac.get("/")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/html")
    assert "/agent/chat" in res.text
