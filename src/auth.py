"""
CliniGuard – Authentication & Rate-Limiting Middleware
======================================================

Provides two lightweight security layers designed for the Streamlit frontend:

1.  **API-Key Gate** (optional):
    Set ACCESS_KEY=<secret> in your .env.  Every session must supply the
    correct key in the sidebar before any query is processed.  If ACCESS_KEY
    is empty the gate is disabled (open-access / development mode).

2.  **Per-Session Rate Limiter**:
    Configurable via RATE_LIMIT_PER_MINUTE env var (default: 10 queries/min).
    Implemented as an in-memory sliding window keyed on session_id.
    Since Streamlit runs single-process this is sufficient for local/small-
    team deployments.  For multi-worker production, swap for Redis.

Usage in Streamlit frontend:
    from src.auth import authenticate_session, check_rate_limit, record_query
    
    if not authenticate_session(session_id, api_key_input):
        st.stop()
    
    allowed, wait_secs = check_rate_limit(session_id)
    if not allowed:
        st.error(f"Rate limit exceeded. Try again in {wait_secs}s.")
        st.stop()
    
    record_query(session_id)  # call after successful rate check
"""

import os
import time
import hashlib
import hmac
import threading
from collections import defaultdict, deque
from typing import Tuple, Dict, Set, Optional
from src.observability import get_logger

logger = get_logger("auth")


# -- Configuration -------------------------------------------------------------
ACCESS_KEY: str = os.getenv("ACCESS_KEY", "").strip()
RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
_WINDOW_SECONDS: int = 60

# -- Thread Synchronization ---------------------------------------------------
# Process-local locks to ensure thread safety across concurrent requests/sessions.
# Note: In-memory state is process-local and does not synchronize across multiple
# worker processes or distributed servers. For multi-worker clusters, use Redis.
_auth_lock: threading.Lock = threading.Lock()
_rate_limit_lock: threading.Lock = threading.Lock()

# -- In-memory rate-limit store ------------------------------------------------
# Maps session_id -> deque of monotonic timestamps of recent queries
_query_timestamps: Dict[str, deque] = defaultdict(deque)

# -- Authenticated sessions cache ----------------------------------------------
# Set of session_ids that have passed the key gate this process lifetime
_authenticated_sessions: Set[str] = set()


# ------------------------------------------------------------------------------
# Authentication helpers
# ------------------------------------------------------------------------------

def is_auth_enabled() -> bool:
    """Returns True when ACCESS_KEY enforcement is active."""
    return bool(ACCESS_KEY)


def _constant_time_compare(a: str, b: str) -> bool:
    """Timing-safe string comparison to prevent timing-based key enumeration."""
    return hmac.compare_digest(
        hashlib.sha256(a.encode()).digest(),
        hashlib.sha256(b.encode()).digest(),
    )


def validate_api_key(provided_key: str) -> bool:
    """
    Stateless API key validation using timing-safe comparison.
    Used for REST API headers where each request must authenticate independently.
    """
    if not is_auth_enabled():
        return True
    if not provided_key or not provided_key.strip():
        return False
    return _constant_time_compare(provided_key.strip(), ACCESS_KEY)


def authenticate_session(session_id: str, provided_key: str) -> bool:
    """
    Validates the provided API key for this session in a thread-safe manner.

    - If ACCESS_KEY is not configured, always returns True (open-access / dev mode).
    - Once a session is authenticated it stays authenticated for its lifetime.
    - Uses timing-safe comparison to prevent oracle attacks.
    - Synchronized with _auth_lock to prevent concurrent mutation of the cache.

    Args:
        session_id:    Unique session identifier.
        provided_key:  Key entered by the user in the sidebar.

    Returns:
        True if the session is authenticated (or auth is disabled).
    """
    if not is_auth_enabled():
        return True

    with _auth_lock:
        if session_id in _authenticated_sessions:
            return True

    # Perform timing-safe cryptographic comparison outside the lock
    if not provided_key or not provided_key.strip():
        return False

    key_valid = _constant_time_compare(provided_key.strip(), ACCESS_KEY)

    if key_valid:
        with _auth_lock:
            _authenticated_sessions.add(session_id)
        logger.info("Session %s successfully authenticated", session_id)
        return True

    logger.warning("Authentication failed for session %s: invalid access key", session_id)
    return False


def deauthenticate_session(session_id: str) -> None:
    """Removes session from the authenticated set (logout) under _auth_lock."""
    with _auth_lock:
        _authenticated_sessions.discard(session_id)


def reset_auth_state() -> None:
    """Resets all in-memory authentication state (for test isolation)."""
    with _auth_lock:
        _authenticated_sessions.clear()


# ------------------------------------------------------------------------------
# Rate limiting helpers
# ------------------------------------------------------------------------------

def check_rate_limit(session_id: str, record: bool = False) -> Tuple[bool, int]:
    """
    Sliding-window rate limiter (in-memory, per session_id).
    Thread-safe implementation synchronized via _rate_limit_lock.

    Args:
        session_id: Unique session identifier.
        record:     If True, atomically appends the query timestamp when allowed.
                    This prevents TOCTOU race conditions under concurrent requests.
                    Defaults to False for backward compatibility with manual record_query calls.

    Returns:
        (allowed: bool, wait_seconds: int)
        wait_seconds > 0 only when allowed is False; it is the number of
        seconds until the oldest query expires from the current window.
    """
    with _rate_limit_lock:
        now = time.monotonic()
        window_start = now - _WINDOW_SECONDS
        q = _query_timestamps[session_id]

        # Evict timestamps outside the sliding window
        while q and q[0] < window_start:
            q.popleft()

        if len(q) >= RATE_LIMIT_PER_MINUTE:
            wait = int(_WINDOW_SECONDS - (now - q[0])) + 1
            logger.warning("Rate limit exceeded for session %s (wait %ds)", session_id, wait)
            return False, wait

        if record:
            q.append(now)

        return True, 0


def check_and_record_rate_limit(session_id: str) -> Tuple[bool, int]:
    """
    Atomically checks rate limit and records query timestamp if allowed.
    Convenience wrapper for check_rate_limit(session_id, record=True).
    """
    return check_rate_limit(session_id, record=True)


def record_query(session_id: str) -> None:
    """
    Records a query timestamp under _rate_limit_lock.
    Must be called after a successful check_rate_limit if record=True was not used.
    """
    with _rate_limit_lock:
        _query_timestamps[session_id].append(time.monotonic())


def get_usage_stats(session_id: str) -> dict:
    """
    Returns current rate-limit usage statistics for sidebar display.
    Thread-safe read and window compaction under _rate_limit_lock.

    Returns dict with:
        queries_in_window: Queries submitted in the last 60 s.
        limit:             Configured queries-per-minute cap.
        remaining:         Remaining queries before being rate-limited.
        window_resets_in:  Seconds until oldest query ages out (0 if window is empty).
    """
    with _rate_limit_lock:
        now = time.monotonic()
        window_start = now - _WINDOW_SECONDS
        q = _query_timestamps[session_id]

        # Evict stale entries to ensure accurate stats
        while q and q[0] < window_start:
            q.popleft()

        count = len(q)
        remaining = max(0, RATE_LIMIT_PER_MINUTE - count)
        resets_in = int(_WINDOW_SECONDS - (now - q[0])) if q else 0

        return {
            "queries_in_window": count,
            "limit": RATE_LIMIT_PER_MINUTE,
            "remaining": remaining,
            "window_resets_in": resets_in,
        }


def reset_rate_limit_state() -> None:
    """Resets all in-memory rate-limiting counters (for test isolation)."""
    with _rate_limit_lock:
        _query_timestamps.clear()
