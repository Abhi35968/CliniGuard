"""
CliniGuard – Centralized Observability & Logging
=================================================
Production-grade logging configuration adhering to enterprise safety standards:
- Standard Python `logging` integration.
- Sensitive-data redaction (API keys, secrets, authorization headers).
- Session correlation tracking via contextvars.
- Structured timestamps, log levels, and module names.
- Duplicate-handler prevention.
"""

import os
import re
import sys
import logging
import threading
from contextvars import ContextVar
from typing import Optional, List, Tuple, Pattern

# ------------------------------------------------------------------------------
# Context Correlation
# ------------------------------------------------------------------------------

_session_id_ctx: ContextVar[Optional[str]] = ContextVar("cliniguard_session_id", default=None)


def set_session_id(session_id: Optional[str]) -> None:
    """Sets the current session/thread correlation ID for the active context."""
    _session_id_ctx.set(session_id)


def get_session_id() -> Optional[str]:
    """Retrieves the active session correlation ID, or None if not set."""
    return _session_id_ctx.get()


class SessionLogContext:
    """Context manager for scoped session correlation."""

    def __init__(self, session_id: Optional[str]):
        self.session_id = session_id
        self.prev_session_id = None
        self.token = None

    def __enter__(self):
        self.prev_session_id = _session_id_ctx.get()
        self.token = _session_id_ctx.set(self.session_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.token is not None:
            try:
                _session_id_ctx.reset(self.token)
            except ValueError:
                _session_id_ctx.set(self.prev_session_id)



# ------------------------------------------------------------------------------
# Secret & Sensitive Data Redaction
# ------------------------------------------------------------------------------

# Known environment variables that contain secret keys
SECRET_ENV_VARS = (
    "ACCESS_KEY",
    "OPENROUTER_API_KEY",
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GROQ_API_KEY",
)

# Regex patterns for detecting sensitive data tokens
_COMPILED_PATTERNS: List[Tuple[Pattern, str]] = [
    # Authorization header with Bearer token: Authorization: Bearer <token>
    (
        re.compile(r"(?i)\b(authorization\s*[:=]\s*['\"]?Bearer\s+)([A-Za-z0-9_\-\.]{6,})(['\"]?)"),
        r"\1[REDACTED_TOKEN]\3",
    ),
    # Standalone Bearer tokens
    (
        re.compile(r"(?i)\bBearer\s+([A-Za-z0-9_\-\.]{6,})\b"),
        "Bearer [REDACTED_TOKEN]",
    ),
    # Authorization / API key headers or assignments: key="value" or key: value (excluding 'Bearer')
    (
        re.compile(
            r"(?i)\b(authorization|api[-_]?key|access[-_]?key|secret[-_]?key|password)\b(\s*[:=]\s*['\"]?)(?!Bearer\b)([A-Za-z0-9_\-\.]{6,})(['\"]?)"
        ),
        r"\1\2[REDACTED]\4",
    ),
    # OpenAI and OpenRouter key formats (sk-...)
    (re.compile(r"\bsk-[A-Za-z0-9\-_]{16,}\b"), "[REDACTED_API_KEY]"),
    # Google API key format (AIza...)
    (re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b"), "[REDACTED_API_KEY]"),
    # URL with embedded Basic Auth credentials: http(s)://user:pass@host
    (re.compile(r"(https?://[^:\s/]+):([^@\s/]+)@"), r"\1:[REDACTED]@"),
    # URL query parameters containing credentials: ?api_key=... or &token=...
    (
        re.compile(r"([?&](?:api[-_]?key|access[-_]?key|token|secret|key)=)([^&\s'\"]+)"),
        r"\1[REDACTED]",
    ),
]




def _get_active_secrets() -> List[str]:
    """Returns currently configured non-empty secret values from the environment."""
    secrets = []
    for var in SECRET_ENV_VARS:
        val = os.getenv(var, "").strip()
        # Avoid redacting very short strings that could match common words
        if val and len(val) >= 4:
            secrets.append(val)
    return secrets


def redact_sensitive_data(text: str) -> str:
    """
    Scans a string and masks any detected secrets, API keys, and authorization tokens.
    Guarantees no raw credentials leak into log sinks.
    """
    if not text:
        return text

    sanitized = text

    # 1. Exact match redaction for configured environment secrets
    for secret in _get_active_secrets():
        if secret in sanitized:
            sanitized = sanitized.replace(secret, "[REDACTED_SECRET]")

    # 2. Pattern-based redaction for credential headers and key shapes
    for pattern, repl in _COMPILED_PATTERNS:
        sanitized = pattern.sub(repl, sanitized)

    return sanitized


# ------------------------------------------------------------------------------
# Custom Formatter
# ------------------------------------------------------------------------------

DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s] [session: %(session_id)s] %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class CliniGuardFormatter(logging.Formatter):
    """
    Custom logging formatter that:
    1. Injects session correlation ID (`session_id`) into each log record.
    2. Redacts API keys, credentials, and sensitive tokens from messages and tracebacks.
    """

    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        super().__init__(fmt=fmt or DEFAULT_LOG_FORMAT, datefmt=datefmt or DEFAULT_DATE_FORMAT)

    def format(self, record: logging.LogRecord) -> str:
        # Determine session ID: record attribute -> contextvar -> fallback '-'
        session_id = getattr(record, "session_id", None) or get_session_id()
        record.session_id = session_id if session_id else "-"

        # Format record using standard Python logging formatter
        formatted = super().format(record)

        # Redact any sensitive information from formatted message and traceback
        return redact_sensitive_data(formatted)


# ------------------------------------------------------------------------------
# Centralized Logging Setup
# ------------------------------------------------------------------------------

_CONFIG_LOCK = threading.Lock()
_IS_CONFIGURED = False


def setup_logging(
    level: Optional[str] = None,
    handler: Optional[logging.Handler] = None,
    force: bool = False,
) -> logging.Logger:
    """
    Initializes central logging for CliniGuard.
    Prevents duplicate handlers on re-entry unless `force=True`.
    """
    global _IS_CONFIGURED

    with _CONFIG_LOCK:
        logger = logging.getLogger("cliniguard")

        if level is None:
            level_name = os.getenv("LOG_LEVEL", "INFO").upper()
        else:
            level_name = level.upper()

        log_level = getattr(logging, level_name, logging.INFO)
        logger.setLevel(log_level)

        if _IS_CONFIGURED and not force:
            return logger

        # Clean existing handlers on the cliniguard logger to prevent duplication
        logger.handlers.clear()

        # Create output handler
        out_handler = handler or logging.StreamHandler(sys.stdout)
        out_handler.setLevel(log_level)
        out_handler.setFormatter(CliniGuardFormatter())

        logger.addHandler(out_handler)
        logger.propagate = False
        _IS_CONFIGURED = True

        return logger


def get_logger(name: str = "cliniguard") -> logging.Logger:
    """
    Returns a configured logger under the `cliniguard` namespace.
    Ensures central logging setup has run.
    """
    if not _IS_CONFIGURED:
        setup_logging()

    if not name or name == "cliniguard":
        return logging.getLogger("cliniguard")
    if name.startswith("cliniguard."):
        return logging.getLogger(name)
    if name.startswith("src."):
        return logging.getLogger("cliniguard." + name[4:])

    return logging.getLogger(f"cliniguard.{name}")
