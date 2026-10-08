"""AWS and Bedrock plumbing.

Validates AWS credentials, including the friendly ``aws sso login`` hint for an
expired session, and builds the Bedrock provider.
"""

import configparser
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from functools import cache
from http import HTTPStatus
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import TokenRetrievalError

logger = logging.getLogger(__name__)


def _log_bedrock_request(http_response, **_: Any) -> None:
    """Log a Bedrock call like httpx does for the HTTP-based providers.

    botocore has no equivalent of httpx's ``HTTP Request: POST ...`` INFO line,
    so this is hooked onto the client's ``after-call`` event.
    """
    status = http_response.status_code
    try:
        reason = HTTPStatus(status).phrase
    except ValueError:
        reason = ""
    logger.info(f'HTTP Request: POST {http_response.url} "HTTP/1.1 {status} {reason}"')


def _sso_cache_key_for_profile(home: Path, profile_name: str) -> str | None:
    """Return the SSO cache key configured for a profile.

    A profile is SSO-based when it sets ``sso_session`` or, for legacy configs,
    ``sso_start_url``. Botocore keys the cached token by the session name when
    present, otherwise by the start URL.

    :param home: User home directory containing ``.aws/config``.
    :param profile_name: Profile name, where ``default`` is the unnamed profile.
    :return: Cache key string, or None if the profile is not SSO-based.
    """
    config_path = home / ".aws" / "config"
    if not config_path.exists():
        return None

    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    section = "default" if profile_name == "default" else f"profile {profile_name}"
    if not cfg.has_section(section):
        return None

    session_name = cfg.get(section, "sso_session", fallback=None)
    start_url = cfg.get(section, "sso_start_url", fallback=None)
    return session_name or start_url


def _expired_sso_message(home: Path, profile_name: str) -> str | None:
    """Return a login hint when a profile's cached SSO token is missing or expired.

    Checking the cache before calling AWS means an expired session produces the
    ``aws sso login`` command instead of a raw botocore TokenRetrievalError.

    :param home: User home directory containing the SSO token cache.
    :param profile_name: Effective profile name (``default`` when unset).
    :return: Message telling the user how to log in, or None if not applicable.
    """
    cache_key = _sso_cache_key_for_profile(home, profile_name)
    if not cache_key:
        return None

    hint = (
        f"AWS SSO session expired for profile '{profile_name}'. "
        f"Run: aws sso login --profile {profile_name}"
    )
    # botocore names the SSO token cache file after the SHA1 of the cache key.
    # This must match botocore exactly; it is a filename, not a security use.
    cache_file = (
        home
        / ".aws"
        / "sso"
        / "cache"
        / (
            hashlib.sha1(cache_key.encode("utf-8"), usedforsecurity=False).hexdigest()
            + ".json"
        )
    )
    if not cache_file.exists():
        return hint

    try:
        cache_data = json.loads(cache_file.read_text())
        expires_at = datetime.fromisoformat(
            cache_data["expiresAt"].replace("Z", "+00:00")
        )
    except (json.JSONDecodeError, ValueError, KeyError):
        return None

    return hint if expires_at < datetime.now(timezone.utc) else None


@cache
def get_aws_config() -> dict[str, Any]:
    """Return AWS configuration dict for boto3 client initialization.

    Searches for AWS profiles and credentials, validates them, and checks for
    SSO token expiration. Uses ``AWS_PROFILE`` and ``AWS_REGION`` env vars if set.

    :return: Dict with optional ``profile_name`` and ``region_name`` keys, suitable for unpacking into boto3 constructors.
    :raises ValueError: On missing, incomplete, or expired credentials.
    """
    home = Path.home()
    aws_config: dict[str, Any] = {}

    # Discover available profiles from credentials and config files
    available_profiles = set()
    for aws_file in [home / ".aws" / "credentials", home / ".aws" / "config"]:
        if aws_file.exists():
            cfg = configparser.ConfigParser()
            cfg.read(aws_file)
            for section in cfg.sections():
                if section.startswith("sso-session "):
                    continue
                name = section[8:] if section.startswith("profile ") else section
                available_profiles.add(name)

    # Use AWS_PROFILE if it exists, otherwise fall back to default credential chain
    profile_name = os.getenv("AWS_PROFILE")
    if profile_name:
        if profile_name in available_profiles:
            aws_config["profile_name"] = profile_name
        else:
            logger.info(
                f"AWS profile '{profile_name}' not found, using default credential chain"
            )
            os.environ.pop("AWS_PROFILE", None)
            profile_name = None

    region = os.getenv("AWS_REGION")
    if region:
        aws_config["region_name"] = region

    effective_profile = profile_name or "default"

    # Detect a missing or expired SSO token before calling AWS so the user gets
    # the login command rather than a raw botocore TokenRetrievalError.
    sso_expired_msg = _expired_sso_message(home, effective_profile)
    if sso_expired_msg:
        raise ValueError(sso_expired_msg)

    session = boto3.Session(**aws_config)

    try:
        credentials = session.get_credentials()

        if not credentials:
            hint = (
                f"Available profiles: {', '.join(available_profiles)}\n"
                if available_profiles
                else ""
            )
            raise ValueError(
                f"No AWS credentials found.\n{hint}"
                "To configure: aws configure\n"
                "Or set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY environment variables"
            )

        if not credentials.access_key or not credentials.secret_key:
            raise ValueError(
                "Incomplete AWS credentials (missing access key or secret key)"
            )

        session.client("sts").get_caller_identity()
    except TokenRetrievalError as e:
        raise ValueError(
            f"AWS SSO session expired for profile '{effective_profile}'. "
            f"Run: aws sso login --profile {effective_profile}"
        ) from e

    return aws_config


def build_bedrock_provider() -> Any:
    """Return a Bedrock provider with validated AWS config and request logging."""
    from pydantic_ai.providers.bedrock import BedrockProvider

    provider = BedrockProvider(**get_aws_config())
    provider.client.meta.events.register(
        "after-call.bedrock-runtime", _log_bedrock_request
    )
    return provider
