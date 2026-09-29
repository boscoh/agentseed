"""Tests for the native Pydantic AI /agent/chat endpoint (offline, no network)."""

import httpx
from fastapi import FastAPI
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelRequest,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel

import agentseed.server as server


def _user_history(text: str) -> bytes:
    """Return a one-message ModelMessage history, as the client would POST it."""
    return ModelMessagesTypeAdapter.dump_json(
        [ModelRequest(parts=[UserPromptPart(content=text)])]
    )


def _test_app() -> FastAPI:
    app = server.create_app()
    app.state.agent = Agent(TestModel(), instructions="test")
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


async def test_root_and_health():
    app = _test_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
        res = await ac.get("/health")
        assert res.status_code == 200
        assert res.json() == {"service": "agentseed", "status": "ok"}


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
    import pytest

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
