"""Logging setup for the agent server.

Rich formatting is installed on the root logger. Library loggers get a level,
and any handlers they installed (e.g. by uvicorn) are dropped so their records
propagate to the root Rich handler instead.
"""

import logging

from rich.console import Console
from rich.logging import RichHandler

# Levels for app and third-party loggers. Pydantic AI delegates to provider
# SDKs that log through different HTTP clients: groq uses httpx, while openai
# (also used for ollama) and anthropic use the httpx2 fork. Both are kept at
# INFO so every provider logs one "HTTP Request: POST ..." line per call;
# Bedrock (botocore) gets the same line from a hook in aws.py.
LOGGER_LEVELS: dict[str, int] = {
    "__main__": logging.INFO,
    "boto3": logging.INFO,
    "uvicorn": logging.INFO,
    "uvicorn.error": logging.INFO,
    "uvicorn.server": logging.INFO,
    "botocore": logging.WARNING,
    "urllib3": logging.WARNING,
    "httpx": logging.INFO,
    "httpx2": logging.INFO,
    "httpcore": logging.WARNING,
    "openai": logging.WARNING,
    "pydantic_ai": logging.WARNING,
    "h11": logging.WARNING,
    "uvicorn.access": logging.WARNING,
}


def setup_logging(level: int | str = logging.INFO) -> None:
    """Install Rich logging on the root logger and set library levels.

    Call as early as possible, before AWS or other service calls.

    :param level: Root log level, as a number or a name like ``"INFO"``.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    rich_handler = RichHandler(
        console=Console(stderr=True),
        log_time_format="[%X]",
    )
    rich_handler.setLevel(level)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)
    root_logger.addHandler(rich_handler)

    for name, logger_level in LOGGER_LEVELS.items():
        logger = logging.getLogger(name)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        logger.setLevel(logger_level)
        # uvicorn sets propagate=False, which would black-hole its records once
        # the handlers above are removed.
        logger.propagate = True
