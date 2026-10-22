"""Logging setup for the agent server.

Rich formatting is installed on the root logger. Library loggers get a level,
and any handlers they installed (e.g. by uvicorn) are dropped so their records
propagate to the root Rich handler instead.

Application code logs through ``log_kv``, which prints an event name and then
its fields as a block of key/value lines. Keys are left-justified, so every
value starts in the same column and a block is quick to scan.

HTTP request lines from the provider HTTP clients (httpx and its httpx2 fork)
and from the uvicorn access log are rewritten into the same block form, with
the event name ``http request`` and the fields method, url, and status.

Colors are the same for every block: the event name is plain, keys are muted grey,
and values have no color. The Rich handler does not highlight numbers, URLs,
or strings on its own.
"""

import logging
from collections.abc import Callable
from typing import Any

from rich.console import Console
from rich.highlighter import NullHighlighter
from rich.logging import RichHandler
from rich.markup import escape

# Levels for app and third-party loggers. Pydantic AI delegates to provider
# SDKs that log through different HTTP clients: groq uses httpx, while openai
# (also used for ollama) and anthropic use the httpx2 fork. Both are kept at
# INFO so every provider logs one HTTP request line per call; that line is
# then rewritten as a key/value block by ``HttpRequestFilter``. Bedrock
# (botocore) gets the same block from a hook in aws.py.
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

# Width of the key column in ``log_kv`` blocks: keys are left-justified to
# this width, so every value starts in the same column.
KEY_WIDTH = 10

# Rich style of the keys in ``log_kv`` blocks: a fixed muted grey, so it looks
# the same in every terminal theme.
KEY_STYLE = "grey62"


def _kv_line(key: str, value: Any) -> str:
    """Return one ``key  value`` line, with the key in the key column."""
    label = f"[{KEY_STYLE}]{escape(key).ljust(KEY_WIDTH)}[/{KEY_STYLE}]"
    return f"{label}  {escape(str(value))}".rstrip()


def format_kv(event: str, fields: dict[str, Any]) -> str:
    """Return an event and its fields as a block of aligned key/value lines.

    The event name comes first. Each field then gets its own line, with the key
    left-justified in a fixed column and the value after it. Example::

        agent build
          model     bedrock:qwen.qwen3-coder

    :param event: Short event name, such as ``"model switch"``.
    :param fields: Fields, printed in the given order, one per line.
    :return: Rich markup text for the log handler.
    """
    lines = [escape(event)]
    lines += [_kv_line(key, value) for key, value in fields.items()]
    return "\n".join(lines)


def format_request(method: str, url: str, **fields: Any) -> str:
    """Return an HTTP request as an ``http request`` event with its fields.

    The method, URL, and any other fields are key/value lines, in the same
    form as every other log block. Example::

        http request
          method    POST
          url       https://api.anthropic.com/v1/messages
          status    HTTP/1.1 200 OK

    :param method: HTTP method, such as ``"POST"``.
    :param url: Request URL.
    :param fields: Further values, printed in the given order, one per line.
    :return: Rich markup text for the log handler.
    """
    return format_kv("http request", {"method": method, "url": url, **fields})


# Loggers whose INFO records describe one finished HTTP request.
HTTP_LOGGERS = ("httpx", "httpx2", "uvicorn.access")

# Message templates of the HTTP request lines, with their ``%`` arguments.
HTTPX_REQUEST = 'HTTP Request: %s %s "%s %d %s"'
UVICORN_ACCESS = '%s - "%s %s HTTP/%s" %d'


class HttpRequestFilter(logging.Filter):
    """Rewrite third-party HTTP request lines as aligned key/value blocks.

    httpx, its httpx2 fork, and the uvicorn access log pass the request parts
    as ``%`` arguments. This filter reads those arguments and replaces the
    message, so a request shows as an ``http request`` event with its fields
    aligned below::

        http request
          method    POST
          url       https://api.anthropic.com/v1/messages
          status    HTTP/1.1 200 OK
    """

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if not isinstance(args, tuple) or len(args) != 5:
            return True
        if record.msg == HTTPX_REQUEST:
            method, url, http_version, code, reason = args
            fields = {"status": f"{http_version} {code} {reason}".strip()}
        elif record.msg == UVICORN_ACCESS:
            client, method, url, http_version, code = args
            fields = {
                "status": f"HTTP/{http_version} {code}",
                "client": client,
            }
        else:
            return True
        record.msg = format_request(method, url, **fields)
        record.args = ()
        record.markup = True
        return True


def log_kv(emit: Callable[..., None], event: str, /, **fields: Any) -> None:
    """Log one event, with each field on its own aligned key/value line.

    Pass a bound logger method so the level comes from the caller::

        log_kv(logger.info, "model switch", previous=previous, current=current)

    :param emit: Bound logger method, such as ``logger.warning``.
    :param event: Short event name.
    :param fields: Field values, printed in the given order, one per line.
    """
    emit(format_kv(event, fields), extra={"markup": True})


def log_request(
    emit: Callable[..., None], method: str, url: str, /, **fields: Any
) -> None:
    """Log one HTTP request, with its status and other fields aligned below.

    :param emit: Bound logger method, such as ``logger.info``.
    :param method: HTTP method, such as ``"POST"``.
    :param url: Request URL.
    :param fields: Further values, printed in the given order, one per line.
    """
    emit(format_request(method, url, **fields), extra={"markup": True})


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
        # Drop the path column so ``log_kv`` lines keep their alignment.
        show_path=False,
        # No automatic highlighting: it colors numbers, URLs, and strings by
        # their shape. Only the markup from ``format_kv`` sets colors.
        highlighter=NullHighlighter(),
        # An empty list turns off the default keyword highlight, which would
        # print GET, POST, and the other HTTP methods in bold yellow.
        keywords=[],
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

    # Rewrite HTTP request lines from the client libraries and the access log.
    # A filter runs on the record before any handler sees it.
    for name in HTTP_LOGGERS:
        logger = logging.getLogger(name)
        logger.filters = [
            item for item in logger.filters if not isinstance(item, HttpRequestFilter)
        ]
        logger.addFilter(HttpRequestFilter())
