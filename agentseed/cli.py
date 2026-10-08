#!/usr/bin/env python3
"""CLI for the agentseed server."""

import logging
import socket
import threading
import time
import webbrowser
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

import cyclopts
from cyclopts import Parameter
from dotenv import load_dotenv

from agentseed.logger import setup_logging

load_dotenv()

logger = logging.getLogger(__name__)

app = cyclopts.App(help="agentseed: minimal agent server with pluggable LLM providers.")


def open_browser_when_ready(url: str, timeout: float = 30.0) -> None:
    """Open ``url`` in the browser once the server accepts connections.

    If the server is not ready after ``timeout`` seconds, open the browser anyway.

    :param url: Server URL to open.
    :param timeout: Seconds to wait for the server.
    """
    parsed = urlparse(url)
    deadline = time.monotonic() + timeout
    ready = False
    while not ready and time.monotonic() < deadline:
        try:
            socket.create_connection((parsed.hostname, parsed.port), timeout=1).close()
            ready = True
        except OSError:
            time.sleep(0.25)
    note = "" if ready else " (server not ready after timeout)"
    logger.info(f"Opening {url} in browser{note}...")
    try:
        webbrowser.open(url)
    except Exception as e:
        logger.error(f"Could not open browser: {e}")


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
    """Run the agentseed server (default command).

    :param host: Bind host address
    :param port: Bind port number
    :param reload: Enable auto-reload for development
    :param open_browser: Open browser when server is ready
    :param ssl: Enable HTTPS with SSL certificates
    :param ssl_cert: Path to SSL certificate file
    :param ssl_key: Path to SSL private key file
    """
    import uvicorn

    from agentseed.server import create_app

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
        # Wait for the server port to accept connections, then open the UI.
        threading.Thread(
            target=open_browser_when_ready, args=(base_url,), daemon=True
        ).start()

    if reload:
        uvicorn.run(
            "agentseed.server:app",
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


def main():
    app()


if __name__ == "__main__":
    main()
