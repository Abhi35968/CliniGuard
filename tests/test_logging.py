"""
CliniGuard – Observability & Logging Test Suite
================================================
Verifies:
1. Central logger configuration and duplicate-handler prevention.
2. Formatter layout (timestamp, log level, logger name, session ID).
3. ContextVar-based session correlation across scopes and threads.
4. Comprehensive redaction of API keys, bearer tokens, passwords, and environment secrets.
5. Formatter redaction over exception tracebacks and formatted records.
"""

import os
import io
import logging
import threading
import pytest

from src.observability.logging_config import (
    setup_logging,
    get_logger,
    set_session_id,
    get_session_id,
    SessionLogContext,
    redact_sensitive_data,
    CliniGuardFormatter,
    DEFAULT_LOG_FORMAT,
)


@pytest.fixture(autouse=True)
def reset_logging_state():
    """Ensures clean logging and correlation state before each test."""
    set_session_id(None)
    yield
    set_session_id(None)


def test_logger_setup_singleton_handlers():
    """Verifies setup_logging does not attach duplicate handlers when invoked repeatedly."""
    stream_buf = io.StringIO()
    handler = logging.StreamHandler(stream_buf)

    # Initial setup
    logger1 = setup_logging(level="INFO", handler=handler, force=True)
    assert len(logger1.handlers) == 1

    # Re-running setup without force should be a no-op regarding handlers
    logger2 = setup_logging(level="INFO")
    assert logger1 is logger2
    assert len(logger1.handlers) == 1


def test_get_logger_namespace_mapping():
    """Verifies hierarchical logger naming under the cliniguard namespace."""
    log_default = get_logger()
    assert log_default.name == "cliniguard"

    log_custom = get_logger("retriever")
    assert log_custom.name == "cliniguard.retriever"

    log_src = get_logger("src.nodes")
    assert log_src.name == "cliniguard.nodes"

    log_prefixed = get_logger("cliniguard.api")
    assert log_prefixed.name == "cliniguard.api"


def test_session_correlation_in_log_output():
    """Verifies that session correlation ID is formatted correctly in log output."""
    stream_buf = io.StringIO()
    handler = logging.StreamHandler(stream_buf)
    logger = setup_logging(level="INFO", handler=handler, force=True)

    # 1. Without session ID -> should show [session: -]
    set_session_id(None)
    logger.info("Message without session")
    output1 = stream_buf.getvalue()
    assert "[session: -]" in output1
    assert "Message without session" in output1

    # Clear buffer
    stream_buf.truncate(0)
    stream_buf.seek(0)

    # 2. With global session ID
    set_session_id("sess-global-456")
    logger.info("Message with global session")
    output2 = stream_buf.getvalue()
    assert "[session: sess-global-456]" in output2
    assert "Message with global session" in output2

    # Clear buffer
    stream_buf.truncate(0)
    stream_buf.seek(0)

    # 3. With scoped SessionLogContext
    with SessionLogContext("sess-scoped-789"):
        logger.info("Message inside scoped context")

    output3 = stream_buf.getvalue()
    assert "[session: sess-scoped-789]" in output3

    # Reset to previous global session
    assert get_session_id() == "sess-global-456"


def test_session_correlation_thread_isolation():
    """Verifies that contextvar session IDs remain isolated across concurrent threads."""
    results = {}

    def worker(session_name):
        with SessionLogContext(session_name):
            results[session_name] = get_session_id()

    t1 = threading.Thread(target=worker, args=("session-thread-A",))
    t2 = threading.Thread(target=worker, args=("session-thread-B",))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert results["session-thread-A"] == "session-thread-A"
    assert results["session-thread-B"] == "session-thread-B"
    assert get_session_id() is None


def test_redact_sensitive_data_patterns():
    """Verifies regex pattern redaction for Bearer tokens, API headers, and key signatures."""
    # Bearer token
    raw_bearer = "Authorization: Bearer my_super_secret_jwt_token_12345"
    redacted_bearer = redact_sensitive_data(raw_bearer)
    assert "my_super_secret_jwt_token_12345" not in redacted_bearer
    assert "Bearer [REDACTED_TOKEN]" in redacted_bearer

    # OpenAI / OpenRouter style key
    raw_key = "Failed request with key sk-abcdef1234567890abcdef123456"
    redacted_key = redact_sensitive_data(raw_key)
    assert "sk-abcdef1234567890abcdef123456" not in redacted_key
    assert "[REDACTED_API_KEY]" in redacted_key

    # Google API key style
    raw_google = "Google GenAI error with key AIzaSyD1234567890abcdefghijklmnopqrstuv"
    redacted_google = redact_sensitive_data(raw_google)
    assert "AIzaSyD1234567890abcdefghijklmnopqrstuv" not in redacted_google
    assert "[REDACTED_API_KEY]" in redacted_google

    # Header / parameter assignment style
    raw_header = "request failed: api_key='secret_pass_xyz987'"
    redacted_header = redact_sensitive_data(raw_header)
    assert "secret_pass_xyz987" not in redacted_header
    assert "api_key='[REDACTED]'" in redacted_header



def test_redact_active_environment_secret(monkeypatch):
    """Verifies that secrets currently defined in environment variables are actively redacted."""
    monkeypatch.setenv("ACCESS_KEY", "cliniguard_super_secret_prod_key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-9999988888777776666655555")

    raw_text = "Received auth check with cliniguard_super_secret_prod_key and sk-or-v1-9999988888777776666655555."
    redacted = redact_sensitive_data(raw_text)

    assert "cliniguard_super_secret_prod_key" not in redacted
    assert "sk-or-v1-9999988888777776666655555" not in redacted
    assert "[REDACTED_SECRET]" in redacted or "[REDACTED_API_KEY]" in redacted


def test_formatter_redacts_exceptions_and_tracebacks(monkeypatch):
    """Verifies that the CliniGuardFormatter redacts secrets even when embedded in exception objects."""
    monkeypatch.setenv("ACCESS_KEY", "secret_database_password_999")

    stream_buf = io.StringIO()
    handler = logging.StreamHandler(stream_buf)
    logger = setup_logging(level="ERROR", handler=handler, force=True)

    try:
        raise ValueError("Connection failed to server with password secret_database_password_999")
    except Exception as e:
        logger.error("Database connection exception", exc_info=True)

    output = stream_buf.getvalue()
    assert "secret_database_password_999" not in output
    assert "[REDACTED_SECRET]" in output
    assert "Database connection exception" in output


def test_redact_credentials_in_urls():
    """Verifies that URLs containing basic-auth credentials or API key query parameters are scrubbed."""
    # Basic auth embedded in URL
    raw_url1 = "Failed to connect to https://admin:super_secret_db_pass_123@cluster.internal:8000/api/v1"
    redacted1 = redact_sensitive_data(raw_url1)
    assert "super_secret_db_pass_123" not in redacted1
    assert "https://admin:[REDACTED]@cluster.internal:8000/api/v1" in redacted1

    # Query param with api_key
    raw_url2 = "Weather service call: https://api.open-meteo.com/v1/forecast?api_key=my_weather_secret_key_456&lat=12.97"
    redacted2 = redact_sensitive_data(raw_url2)
    assert "my_weather_secret_key_456" not in redacted2
    assert "?api_key=[REDACTED]&lat=12.97" in redacted2

    # Query param with token
    raw_url3 = "Webhook delivery failed: https://hooks.slack.com/services?token=xoxb-secret-slack-token-789"
    redacted3 = redact_sensitive_data(raw_url3)
    assert "xoxb-secret-slack-token-789" not in redacted3
    assert "?token=[REDACTED]" in redacted3


def test_redact_exception_message_without_traceback(monkeypatch):
    """Verifies that inline exception messages without stack traces are also redacted."""
    stream_buf = io.StringIO()
    handler = logging.StreamHandler(stream_buf)
    logger = setup_logging(level="WARNING", handler=handler, force=True)

    err = RuntimeError("Upstream API returned 401 with api_key='secret_key_in_err_msg'")
    logger.warning("Upstream call failed: %s", err)

    output = stream_buf.getvalue()
    assert "secret_key_in_err_msg" not in output
    assert "api_key='[REDACTED]'" in output

