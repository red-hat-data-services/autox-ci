"""Unit tests: credentials must never reach pytest output or logs."""

from __future__ import annotations

import logging
import sys

from autox_tests.lib.redaction import (
    REDACTED,
    SecretStr,
    install_log_redaction,
    redact_report,
    redact_text,
    register_secret,
    restore_log_redaction,
)

SECRET = "sha256~unit-test-secret-value"


class _StubReport:
    """Mimics ``_pytest.reports.TestReport``: captured views derive from sections."""

    def __init__(self, secret: str) -> None:
        self.longrepr = f"automl_functional_config = {{'rhoai_token': '{secret}'}}"
        self.sections = [
            ("Captured stdout call", f"printed token {secret}"),
            ("Captured log call", f"INFO token={secret}"),
        ]

    @property
    def capstdout(self) -> str:
        return "".join(
            content
            for title, content in self.sections
            if title == "Captured stdout call"
        )

    @property
    def caplog(self) -> str:
        return "".join(
            content for title, content in self.sections if title == "Captured log call"
        )


def test_redact_text_removes_registered_secret() -> None:
    register_secret(SECRET)

    assert SECRET not in redact_text(f"config={{'rhoai_token': '{SECRET}'}}")
    assert REDACTED in redact_text(SECRET)


def test_redact_text_uses_environment(monkeypatch) -> None:
    monkeypatch.setenv("RHOAI_TOKEN", SECRET)

    assert redact_text(f"Bearer {SECRET}") == f"Bearer {REDACTED}"


def test_redact_text_handles_pytest_truncation(monkeypatch) -> None:
    secret = "sha256~abcdefghijklmnopqrstuvwxyz0123456789"
    monkeypatch.setenv("RHOAI_TOKEN", secret)
    # pytest renders long reprs as head + "..." + tail.
    truncated = f"{{'rhoai_token': '{secret[:12]}...{secret[-12:]}'}}"

    redacted = redact_text(truncated)

    assert secret[:12] not in redacted
    assert secret[-12:] not in redacted
    assert REDACTED in redacted


def test_redact_text_ignores_short_values(monkeypatch) -> None:
    monkeypatch.setenv("RHOAI_TOKEN", "token")

    assert redact_text("a token appears here") == "a token appears here"


def test_redact_report_scrubs_every_channel() -> None:
    register_secret(SECRET)
    report = _StubReport(SECRET)

    redact_report(report)

    assert isinstance(report.longrepr, str)
    assert SECRET not in report.longrepr
    assert SECRET not in report.capstdout
    assert SECRET not in report.caplog
    assert SECRET not in report.sections[0][1]


def test_secret_str_masks_repr_only() -> None:
    secret = SecretStr(SECRET)

    assert repr(secret) == f"'{REDACTED}'"
    assert repr({"rhoai_token": secret}) == f"{{'rhoai_token': '{REDACTED}'}}"
    assert str(secret) == SECRET
    assert secret == SECRET


def test_install_log_redaction_is_idempotent() -> None:
    install_log_redaction()
    installed = logging.getLogRecordFactory()

    install_log_redaction()

    assert logging.getLogRecordFactory() is installed


def test_install_log_redaction_scrubs_new_records(monkeypatch) -> None:
    monkeypatch.setenv("RHOAI_TOKEN", SECRET)
    install_log_redaction()

    record = logging.getLogRecordFactory()(
        "test", logging.INFO, __file__, 1, "token=%s", (SECRET,), None
    )

    assert SECRET not in record.getMessage()
    assert REDACTED in record.getMessage()


def test_redact_text_handles_head_only_truncation(monkeypatch) -> None:
    """pytest prints long fixture values as ``head...`` and drops the tail."""
    monkeypatch.setenv("RHOAI_TOKEN", SECRET)
    leaked = f"{{'rhoai_token': '{SECRET[:20]}...'}}"

    redacted = redact_text(leaked)

    assert SECRET[:20] not in redacted
    assert SECRET[:12] not in redacted
    assert REDACTED in redacted


def test_redact_text_preserves_unrelated_text(monkeypatch) -> None:
    monkeypatch.setenv("RHOAI_TOKEN", SECRET)
    text = f"head={SECRET[:16]} KEEP_THIS_MIDDLE_MESSAGE tail={SECRET[-16:]}"

    redacted = redact_text(text)

    assert "KEEP_THIS_MIDDLE_MESSAGE" in redacted
    assert SECRET[:16] not in redacted
    assert SECRET[-16:] not in redacted


def test_redact_report_scrubs_truncated_config_repr(monkeypatch) -> None:
    """Reproduces the leaking ``automl_functional_config = {...}`` failure line."""
    monkeypatch.setenv("RHOAI_TOKEN", SECRET)
    report = _StubReport(SECRET)
    report.longrepr = (
        "automl_functional_config = {'rhoai_url': 'https://api', "
        f"'rhoai_token': '{SECRET[:20]}...', "
        "'rhoai_project': 'automl-integration-test'}}"
    )

    redact_report(report)

    assert SECRET[:12] not in report.longrepr
    assert "automl-integration-test" in report.longrepr


def test_log_record_exc_info_is_redacted(monkeypatch) -> None:
    monkeypatch.setenv("RHOAI_TOKEN", SECRET)
    install_log_redaction()
    try:
        raise ValueError(f"auth failed token={SECRET}")
    except ValueError:
        record = logging.getLogRecordFactory()(
            "test", logging.ERROR, __file__, 1, "task failed", (), sys.exc_info()
        )

    rendered = logging.Formatter("%(message)s").format(record)

    assert SECRET not in rendered
    assert REDACTED in rendered


def test_redact_text_redacts_escaped_repr(monkeypatch) -> None:
    """``repr`` doubles backslashes; the escaped form must be scrubbed too."""
    secret = "p@ss\\word!"
    monkeypatch.setenv("NEO4J_PASSWORD", secret)
    escaped = f"password={secret!r}"
    assert "\\\\" in escaped  # sanity: repr doubled the backslash

    redacted = redact_text(escaped)

    assert secret not in redacted
    assert secret.replace("\\", "\\\\") not in redacted
    assert REDACTED in redacted


def test_redact_text_keeps_raw_value_redaction(monkeypatch) -> None:
    secret = "p@ss\\word!"
    monkeypatch.setenv("NEO4J_PASSWORD", secret)

    redacted = redact_text(f"password={secret}")

    assert secret not in redacted
    assert REDACTED in redacted


def test_restore_log_redaction_uninstalls_factory() -> None:
    install_log_redaction()
    assert getattr(logging.getLogRecordFactory(), "_autox_secret_redaction", False)

    restore_log_redaction()
    assert not getattr(logging.getLogRecordFactory(), "_autox_secret_redaction", False)

    # Leave the session-wide redaction installed for the remaining tests.
    install_log_redaction()
