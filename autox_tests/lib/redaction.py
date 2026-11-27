"""Redact credentials from pytest reports, logs, and captured output."""

from __future__ import annotations

import logging
import os
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
# pytest ellipsizes long values in reprs very aggressively, usually as
# ``value[:N] + "..."`` with the tail dropped. Any contiguous prefix or suffix
# of a secret at least this long is treated as a partial leak and redacted.
MIN_PARTIAL_LENGTH = 12

_registered: set[str] = set()
_previous_factory: Any = None


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


def _escaped_variants(secret: str) -> tuple[str, ...]:
    """Return repr-style escaped spellings of ``secret`` drawn from output.

    ``repr`` (and ``print(repr(...))``) renders characters like backslashes as
    doubled sequences, so a raw ``p@ss\\word!`` shows up as ``p@ss\\\\word!``.
    The escaped text no longer contains the raw value, so it must be scrubbed
    separately.
    """
    variants: list[str] = []
    repr_body = repr(secret)[1:-1]
    if repr_body != secret:
        variants.append(repr_body)
    doubled = secret.replace("\\", "\\\\")
    if doubled != secret and doubled not in variants:
        variants.append(doubled)
    return tuple(variants)


def _redact_secret(text: str, secret: str) -> str:
    """Replace ``secret`` and any recognizable partial form of it in ``text``."""
    text = text.replace(secret, REDACTED)
    for variant in _escaped_variants(secret):
        text = text.replace(variant, REDACTED)
    if len(secret) <= MIN_PARTIAL_LENGTH:
        return text

    # pytest truncates long reprs asymmetrically (``head...``), so scrub every
    # leading fragment that still carries at least ``MIN_PARTIAL_LENGTH`` chars.
    if secret[:MIN_PARTIAL_LENGTH] in text:
        for end in range(len(secret) - 1, MIN_PARTIAL_LENGTH - 1, -1):
            fragment = secret[:end]
            if fragment in text:
                text = text.replace(fragment, REDACTED)

    # Some tools keep a trailing fragment (``...tail``); scrub those too.
    if secret[-MIN_PARTIAL_LENGTH:] in text:
        for start in range(1, len(secret) - MIN_PARTIAL_LENGTH + 1):
            fragment = secret[start:]
            if fragment in text:
                text = text.replace(fragment, REDACTED)
    return text


def redact_text(text: str) -> str:
    """Replace every known credential (or recognizable fragment) in ``text``."""
    for secret in known_secrets():
        text = _redact_secret(text, secret)
    return text


def redact_report(report: Any) -> None:
    """Scrub credentials inside a pytest ``TestReport`` in place.

    pytest renders failing-test arguments, locals, captured output, and log
    records into the report; rewriting them here keeps the token out of the
    terminal output and the JUnit XML while leaving the rest untouched.
    ``longreprtext``/``capstdout``/``capstderr``/``caplog`` are derived from
    ``longrepr`` and ``sections`` on access, so scrubbing those covers them too.
    Skipped reports store ``longrepr`` as a ``(path, lineno, reason)`` tuple that
    the JUnit writer and the ``-rs`` summary assert on, so only the reason is
    redacted there and the tuple shape is preserved.
    """
    longrepr = getattr(report, "longrepr", None)
    if (
        isinstance(longrepr, tuple)
        and len(longrepr) == 3
        and isinstance(longrepr[2], str)
    ):
        path, lineno, reason = longrepr
        redacted_reason = redact_text(reason)
        if redacted_reason != reason:
            report.longrepr = (path, lineno, redacted_reason)
    elif longrepr is not None:
        text = longrepr if isinstance(longrepr, str) else str(longrepr)
        redacted = redact_text(text)
        if redacted != text:
            report.longrepr = redacted

    sections = getattr(report, "sections", None)
    if sections:
        report.sections = [(title, redact_text(content)) for title, content in sections]

    # Drop any cached rendered views (defensive; pytest computes them lazily).
    for attr in ("longreprtext", "caplog", "capstdout", "capstderr"):
        getattr(report, "__dict__", {}).pop(attr, None)


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
    # Tracebacks carry the creds too when ``logger.exception`` is used; rewrite
    # the rendered text and drop exc_info so the formatter cannot re-render it.
    if record.exc_info:
        exc_text = record.exc_text or logging.Formatter().formatException(
            record.exc_info
        )
        record.exc_text = redact_text(exc_text)
        record.exc_info = None
    if record.stack_info:
        record.stack_info = redact_text(record.stack_info)
    return record


def install_log_redaction() -> None:
    """Make every log record pass through credential redaction.

    A record factory is used instead of handler filters because pytest creates
    its live-log and capture handlers after ``pytest_configure`` has run.
    """
    global _previous_factory
    factory = logging.getLogRecordFactory()
    if getattr(factory, "_autox_secret_redaction", False):
        return

    def redacting_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        return _redact_record(factory(*args, **kwargs))

    redacting_factory._autox_secret_redaction = True  # type: ignore[attr-defined]
    _previous_factory = factory
    logging.setLogRecordFactory(redacting_factory)


def restore_log_redaction() -> None:
    """Undo :func:`install_log_redaction` if our factory is still installed."""
    global _previous_factory
    factory = logging.getLogRecordFactory()
    if (
        getattr(factory, "_autox_secret_redaction", False)
        and _previous_factory is not None
    ):
        logging.setLogRecordFactory(_previous_factory)
        _previous_factory = None
