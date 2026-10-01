# agentseed

A skeleton for running a [Pydantic AI](https://ai.pydantic.dev) agent inside a
FastAPI server. Fork it as a starting point: the agent, the API and a chat UI are
already wired together, with swappable providers (AWS Bedrock, OpenAI, Anthropic,
Groq, Ollama) behind one async interface.

- **Backend** — FastAPI app hosting a Pydantic AI agent, speaking native Pydantic
  AI message history over HTTP.
- **Frontend** — `agentseed/index.html`, a single HTML file that is the smallest,
  most robust frontend we could make: no build step, no node_modules, just CDN
  scripts and one file to edit.
- **AWS wrapper** — a thin layer over boto3 credentials so Bedrock works the same
  on a laptop with SSO and in a deployed container (see [AWS credentials](#aws-credentials)).

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

uv run agentseed
# http://localhost:8000
```

## Endpoints

- `GET /` — chat UI (`agentseed/index.html`)
- `GET /config` — default provider/model, availability, and all selectable `models` pairs
- `POST /agent/chat[?service=&model=]` — native Pydantic AI message history (ModelMessage JSON arrays); optional query params pick a pair from `models.json`

## AWS credentials

We lean on Pydantic AI for the models, but AWS profiles don't work well out of
the box. The common scenario is:

1. **Local development** — you authenticate with `aws sso login` and pick a
   profile via `AWS_PROFILE`.
2. **Deployment** to EC2, ECS or Kubernetes — there is no `~/.aws` directory and no
   SSO profile; credentials come from the instance role, task role or IRSA via
   the default credential chain.

Passing a profile straight to boto3 breaks one side or the other: a profile
name that doesn't exist on the server raises `ProfileNotFound`, and an expired
SSO token locally surfaces as an opaque `TokenRetrievalError`.

`get_aws_config()` in `agentseed/providers.py` smooths this over and returns
kwargs for `BedrockProvider` / boto3:

- If `AWS_PROFILE` is set and exists in `~/.aws/config` or `~/.aws/credentials`,
  it is used. If it doesn't exist (e.g. in a container), it is dropped with a
  warning and the default credential chain is used instead, so the same
  env/config can ship unchanged.
- `AWS_REGION` is passed through as `region_name`.
- SSO tokens are checked in `~/.aws/sso/cache` *before* calling AWS, so an
  expired session gives you the exact fix: `aws sso login --profile <name>`.
- Missing or incomplete credentials raise a `ValueError` listing available
  profiles and how to configure them.

Use `get_aws_config()` anywhere you create a boto3 client, e.g.
`boto3.client("s3", **get_aws_config())`, to get the same behaviour.

## Frontend

The chat UI is a single file, `agentseed/index.html`, based on the one-page app
template from [onepageapp](https://github.com/boscoh/onepageapp). FastAPI serves
it at `/`. It loads Vue 3, Bootstrap and marked from CDNs, and the Vue template
is plain HTML inside `#app`, so there is no build step: edit the file and reload
the page.

It keeps the conversation in `localStorage` as Pydantic AI `ModelMessage` JSON,
and has a provider/model dropdown fed by `/config`. Providers that failed the
startup probe are shown disabled, with the error on hover.

## Layout

- `agentseed/providers.py` — Pydantic AI model/embedding builders, `CHAT_SERVICE`
  selection (default `bedrock`), and `get_aws_config`
- `agentseed/server.py` — FastAPI app (`/`, `/config`, `/agent/chat`)
  and the Pydantic AI agent builder
- `agentseed/cli.py` — `agentseed` server command (cyclopts + uvicorn)
- `agentseed/logger.py` — Rich logging setup
- `agentseed/models.json` — selectable chat and embedding models
- `agentseed/index.html` — single-file chat UI (Vue 3 + Bootstrap via CDN)
