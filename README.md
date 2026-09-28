# minagent

Minimal pluggable LLM agent server: FastAPI + Pydantic AI with swappable
providers (AWS Bedrock, OpenAI, Anthropic, Groq, Ollama) behind one async interface.

## Quick Start

```bash
uv sync

# Configure a provider (bedrock is the default)
export AWS_PROFILE=my-profile          # Bedrock (SSO)
export AWS_REGION=ap-southeast-2       # Bedrock
aws sso login --profile my-profile
# export CHAT_SERVICE=openai|anthropic|groq|ollama   # override provider
export OPENAI_API_KEY=sk-...           # OpenAI
export ANTHROPIC_API_KEY=sk-ant-...    # Anthropic
export GROQ_API_KEY=gsk_...            # Groq

uv run minagent
# http://localhost:8000

uv run minagent check             # which providers are live?
```

## Endpoints

- `GET /` — chat UI (`minagent/index.html`)
- `GET /health` — liveness
- `GET /providers/status[?service=&all_models=]` — live-probe providers (a few tokens each)
- `GET /config` — default provider/model, availability, and all selectable `models` pairs
- `POST /agent/chat[?service=&model=]` — native Pydantic AI message history (ModelMessage JSON arrays); optional query params pick a pair from `models.json`

## Frontend

The chat UI is a single file, `minagent/index.html` (adapted from
chatboti), served by FastAPI at `/`. It loads Vue 3, Bootstrap, marked and pug
from CDNs, so there is no build step: edit the file and reload the page.

It keeps the conversation in `localStorage` as Pydantic AI `ModelMessage` JSON,
and has a provider/model dropdown fed by `/config`. Providers that failed the
startup probe are shown disabled, with the error on hover.

## Layout

- `minagent/providers.py` — Pydantic AI model/embedding builders, `CHAT_SERVICE`
  selection (default `bedrock`), and `get_aws_config`
- `minagent/server.py` — FastAPI app (`/`, `/health`, `/config`, `/agent/chat`)
  and the Pydantic AI agent builder
- `minagent/cli.py` — `minagent` (serve, default) and `minagent check` (cyclopts + uvicorn)
- `minagent/logger.py` — Rich logging setup
- `minagent/models.json` — selectable chat and embedding models
- `minagent/index.html` — single-file chat UI (Vue 3 + Bootstrap via CDN)
- `tests/` — offline tests for the chat endpoint

## Checking providers

`minagent check` sends each provider's default model a one-word prompt
(capped at 16 output tokens) and prints a table of live/failed providers with
latency and the error. It exits non-zero if any probe fails.

```bash
uv run minagent check                 # default model of every provider
uv run minagent check anthropic groq  # only these providers
uv run minagent check --all           # every model in models.json
```

The same probe is available at `GET /providers/status`.

## Tests

```bash
uv run pytest
```
