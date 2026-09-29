#!/usr/bin/env python3
"""CLI for the minagent server."""

import threading
from pathlib import Path
from typing import Annotated

import cyclopts
from cyclopts import Parameter
from dotenv import load_dotenv

from minagent.logger import setup_logging

load_dotenv()

app = cyclopts.App(help="minagent: minimal agent server with pluggable LLM providers.")


@app.default
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: Annotated[bool, Parameter(name=["--reload", "-r"])] = True,
    open_browser: Annotated[bool, Parameter(name=["--open-browser", "-o"])] = True,
    ssl: Annotated[bool, Parameter(name=["--ssl", "-s"])] = False,
    ssl_cert: str = "cert.pem",
    ssl_key: str = "key.pem",
):
    """Run the minagent server (default command).

    :param host: Bind host address
    :param port: Bind port number
    :param reload: Enable auto-reload for development
    :param open_browser: Open browser when server is ready
    :param ssl: Enable HTTPS with SSL certificates
    :param ssl_cert: Path to SSL certificate file
    :param ssl_key: Path to SSL private key file
    """
    import uvicorn

    from minagent.server import create_app, wait_and_open_browser

    setup_logging()

    # Prepare SSL configuration
    ssl_keyfile = None
    ssl_certfile = None
    if ssl:
        ssl_key_path = Path(ssl_key)
        ssl_cert_path = Path(ssl_cert)
        if not ssl_key_path.exists():
            raise SystemExit(f"SSL key file not found: {ssl_key}")
        if not ssl_cert_path.exists():
            raise SystemExit(f"SSL certificate file not found: {ssl_cert}")
        ssl_keyfile = str(ssl_key_path)
        ssl_certfile = str(ssl_cert_path)

    if open_browser:
        protocol = "https" if ssl else "http"
        # Always use localhost for browser opening, regardless of bind host
        browser_host = "localhost" if host in ("0.0.0.0", "127.0.0.1", "localhost") else host
        base_url = f"{protocol}://{browser_host}:{port}"
        # Poll the always-present health endpoint, then open the UI root.
        thread = threading.Thread(
            target=wait_and_open_browser,
            args=(f"{base_url}/health", base_url),
            daemon=True,
        )
        thread.start()

    if reload:
        uvicorn.run(
            "minagent.server:app",
            host=host,
            port=port,
            log_config=None,
            reload=True,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
        )
    else:
        uvicorn.run(
            create_app(),
            host=host,
            port=port,
            log_config=None,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
        )


@app.command
def check(
    *services: str,
    all_models: Annotated[bool, Parameter(name=["--all", "-a"])] = False,
    timeout: float = 20.0,
):
    """Check which providers are live by sending each a tiny request.

    :param services: Providers to check (default: all in models.json).
    :param all_models: Check every configured model, not just each default.
    :param timeout: Per-probe timeout in seconds.
    """
    import asyncio
    import logging

    from rich.console import Console
    from rich.table import Table

    from minagent.providers import check_models

    setup_logging(logging.WARNING)
    console = Console()
    with console.status("Probing providers..."):
        results = asyncio.run(check_models(list(services) or None, all_models, timeout))

    table = Table("", "service", "model", "latency", "error")
    for r in results:
        table.add_row(
            "[green]✓[/]" if r["live"] else "[red]✗[/]",
            r["service"],
            r["model"],
            f"{r['latency_ms']} ms",
            r["error"] or "",
        )
    console.print(table)
    live = sum(r["live"] for r in results)
    console.print(f"{live}/{len(results)} live")
    if live < len(results):
        raise SystemExit(1)


def main():
    app()


if __name__ == "__main__":
    main()
