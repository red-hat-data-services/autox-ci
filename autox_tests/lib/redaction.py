"""Redact credentials from pytest reports, logs, and captured output."""

from __future__ import annotations

import logging
import os
import re
from typing import Any

REDACTED = "***REDACTED***"

# Environment-provided credentials are scrubbed even if a test never touched
# the config helpers, so pytest tracebacks cannot echo them.
SECRET_ENV_VARS = (
    "RHOAI_TOKEN",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "ARTIFACTS_AWS_ACCESS_KEY_ID",
    "ARTIFACTS_AWS_SECRET_ACCESS_KEY",
    "RHOAI_KSERVE_STORAGE_KEY",
    "MAAS_API_KEY",
    "NEO4J_PASSWORD",
)

# Short values (for example "token" in unit tests) would mangle unrelated text.
MIN_SECRET_LENGTH = 6
# pytest truncates long reprs as ``head + "..." + tail``; redact that shape too.
FRAGMENT_LENGTH = 8
FRAGMENT_WINDOW = 200

_registered: set[str] = set()


def register_secret(value: Any) -> None:
    """Remember a credential so it can be scrubbed from captured output."""
    text = str(value or "").strip()
    if len(text) >= MIN_SECRET_LENGTH:
        _registered.add(text)


def known_secrets() -> tuple[str, ...]:
    """Return registered and environment-provided credentials, longest first."""
    values = set(_registered)
    for name in SECRET_ENV_VARS:
        text = (os.environ.get(name) or "").strip()
        if len(text) >= MIN_SECRET_LENGTH:
            values.add(text)
    return tuple(sorted(values, key=len, reverse=True))


def redact_text(text: str) -> str:
    """Replace every known credential in ``text`` with :data:`REDACTED`.

    pytest ellipsizes long reprs (``head + "..." + tail``), so a truncated
    credential must be matched by its head and tail fragments as well.
    """
    for secret in known_secrets():
        if secret in text:
            text = text.replace(secret, REDACTED)
        if len(secret) >= 2 * FRAGMENT_LENGTH:
            pattern = (
                re.escape(secret[:FRAGMENT_LENGTH])
                + rf"[^\n]{{0,{FRAGMENT_WINDOW}}}?"
                + re.escape(secret[-FRAGMENT_LENGTH:])
            )
            text = re.sub(pattern, REDACTED, text)
    return text


def redact_report(report: Any) -> None:
    """Scrub credentials inside a pytest ``TestReport`` in place.

    pytest renders failing-test arguments, locals, captured output, and log
    records into the report; rewriting them here keeps the token out of the
    terminal output and the JUnit XML while leaving the rest untouched.
    ``capstdout``/``capstderr``/``caplog`` are read-only views over
    ``sections``, so scrubbing the sections covers them too.
    """
    longrepr = getattr(report, "longrepr", None)
    if longrepr is not None:
        text = longrepr if isinstance(longrepr, str) else str(longrepr)
        redacted = redact_text(text)
        if redacted != text:
            report.longrepr = redacted

    sections = getattr(report, "sections", None)
    if sections:
        report.sections = [(title, redact_text(content)) for title, content in sections]


class SecretStr(str):
    """A string whose repr is masked, so dicts and tracebacks never show it.

    ``str()``, formatting, JSON, and equality still return the real value;
    only ``repr()`` is redacted. Use it for credentials kept in config dicts.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return f"'{REDACTED}'" if self else "''"


def _redact_record(record: logging.LogRecord) -> logging.LogRecord:
    try:
        message = record.getMessage()
    except Exception:  # pragma: no cover - never break logging on a bad record
        return record
    redacted = redact_text(message)
    if redacted != message:
        record.msg = redacted
        record.args = ()
    return record


def install_log_redaction() -> None:
    """Make every log record pass through credential redaction.

    A record factory is used instead of handler filters because pytest creates
    its live-log and capture handlers after ``pytest_configure`` has run.
    """
    factory = logging.getLogRecordFactory()
    if getattr(factory, "_autox_secret_redaction", False):
        return

    def redacting_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        return _redact_record(factory(*args, **kwargs))

    redacting_factory._autox_secret_redaction = True  # type: ignore[attr-defined]
    logging.setLogRecordFactory(redacting_factory)
