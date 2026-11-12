"""
CliniGuard Concurrency & Thread-Safety Tests
=============================================
Regression and concurrency tests verifying Finding 6:
1. Concurrent requests cannot corrupt rate-limit counters or bypass limits via race conditions.
2. Authentication state remains consistent under concurrent read/write/discard operations.
3. Locks are not held during slow external operations (LLM / agent execution).
4. Existing rate-limit thresholds, wait calculations, and HTTP responses remain unchanged.
5. Stateless API key validation works consistently across concurrent requests.
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import src.auth as auth
from src.auth import (
    authenticate_session,
    deauthenticate_session,
    check_rate_limit,
    check_and_record_rate_limit,
    record_query,
    get_usage_stats,
    validate_api_key,
    reset_auth_state,
    reset_rate_limit_state,
    _auth_lock,
    _rate_limit_lock,
)
from src.api import app


@pytest.fixture(autouse=True)
def clean_auth_and_ratelimit():
    """Ensure clean in-memory state before and after each test."""
    reset_auth_state()
    reset_rate_limit_state()
    yield
    reset_auth_state()
    reset_rate_limit_state()


def test_concurrent_rate_limiting_prevents_bypass_and_corruption(monkeypatch):
    """
    Test 1: 20 concurrent threads attempt to acquire rate limit simultaneously with limit=5.
    Verifies that atomic check-and-record prevents TOCTOU bypass and avoids counter corruption.
    """
    limit = 5
    num_threads = 20
    monkeypatch.setattr(auth, "RATE_LIMIT_PER_MINUTE", limit)
    session_id = "test-concurrent-rl-session"

    barrier = threading.Barrier(num_threads)
    results = []
    lock = threading.Lock()

    def worker():
        # Synchronize all threads so they hit the rate limiter at the exact same moment
        barrier.wait()
        allowed, wait = check_rate_limit(session_id, record=True)
        with lock:
            results.append((allowed, wait))

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    allowed_count = sum(1 for allowed, _ in results if allowed is True)
    rejected_count = sum(1 for allowed, _ in results if allowed is False)

    # Exactly `limit` requests must be permitted; all excess must be rejected
    assert allowed_count == limit, f"Expected exactly {limit} allowed, got {allowed_count}"
    assert rejected_count == num_threads - limit

    # All rejected calls must indicate positive wait seconds
    for allowed, wait in results:
        if not allowed:
            assert wait > 0, "Rejected request must have wait_seconds > 0"

    # Verify underlying store state and usage statistics
    stats = get_usage_stats(session_id)
    assert stats["queries_in_window"] == limit
    assert stats["remaining"] == 0
    assert stats["limit"] == limit


def test_concurrent_api_requests_enforce_rate_limits(monkeypatch):
    """
    Test 2: Concurrent requests hitting the FastAPI endpoint with limit=4.
    Verifies that exactly 4 return 200 OK and remaining return 429 Too Many Requests.
    """
    limit = 4
    num_requests = 12
    monkeypatch.setattr(auth, "RATE_LIMIT_PER_MINUTE", limit)
    session_id = "test-concurrent-api-session"
    client = TestClient(app)

    barrier = threading.Barrier(num_requests)

    # Mock run_agent_turn so endpoint execution is lightweight and reliable
    with patch("src.api.run_agent_turn") as mock_turn:
        mock_turn.return_value = {
            "final_response": "Advisory mock response",
            "risk_assessment": {"is_safe": True, "risk_level": "LOW"},
            "user_context": {"location": "Kolkata"},
            "weather_data": {"temperature": 25.0},
            "matched_sops": ["SOP-LEISURE-001"],
            "retrieved_sops": [],
        }

        def send_request():
            barrier.wait()
            resp = client.post(
                "/api/v1/query",
                json={"query": "Is it safe to run in Kolkata?", "thread_id": session_id},
            )
            return resp.status_code, resp.json()

        with ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [executor.submit(send_request) for _ in range(num_requests)]
            responses = [f.result() for f in futures]

    status_codes = [status for status, _ in responses]
    successes = status_codes.count(200)
    rate_limited = status_codes.count(429)

    assert successes == limit, f"Expected {limit} 200 responses, got {successes}"
    assert rate_limited == num_requests - limit, f"Expected {num_requests - limit} 429 responses, got {rate_limited}"

    # Verify 429 response structure
    for status, data in responses:
        if status == 429:
            assert "Rate limit exceeded" in data["detail"]


def test_concurrent_authentication_consistency(monkeypatch):
    """
    Test 3: Concurrent authentication attempts with valid and invalid keys.
    Verifies thread safety of _authenticated_sessions under simultaneous read/add/discard.
    """
    valid_key = "secure-concurrency-token-xyz"
    monkeypatch.setattr(auth, "ACCESS_KEY", valid_key)

    valid_session = "session-valid-concurrency"
    invalid_session = "session-invalid-concurrency"
    num_threads = 16

    barrier = threading.Barrier(num_threads)
    results = []
    res_lock = threading.Lock()

    def auth_worker(thread_idx: int):
        barrier.wait()
        if thread_idx % 2 == 0:
            res = authenticate_session(valid_session, valid_key)
            with res_lock:
                results.append(("valid", res))
        else:
            res = authenticate_session(invalid_session, "wrong-key")
            with res_lock:
                results.append(("invalid", res))

    threads = [threading.Thread(target=auth_worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All valid key calls must succeed; all invalid must fail
    for label, res in results:
        if label == "valid":
            assert res is True
        else:
            assert res is False

    # Check cache status
    assert authenticate_session(valid_session, "any-dummy-now") is True
    assert authenticate_session(invalid_session, "wrong-key") is False

    # Test concurrent deauthentication
    deauth_barrier = threading.Barrier(5)

    def deauth_worker():
        deauth_barrier.wait()
        deauthenticate_session(valid_session)

    deauth_threads = [threading.Thread(target=deauth_worker) for _ in range(5)]
    for t in deauth_threads:
        t.start()
    for t in deauth_threads:
        t.join()

    # After deauth, invalid key must be rejected again
    assert authenticate_session(valid_session, "wrong-key") is False


def test_locks_not_held_during_slow_external_operations(monkeypatch):
    """
    Test 4: Verify that locks are not held while slow external operations (e.g. LLM / agent turn) execute.
    Other sessions/threads must be able to acquire rate limits and authenticate without waiting for the slow turn.
    """
    client = TestClient(app)
    agent_started = threading.Event()
    release_agent = threading.Event()

    def slow_agent_turn(*args, **kwargs):
        agent_started.set()
        # Simulate long-running I/O or LLM evaluation (up to 2 seconds unless unblocked)
        release_agent.wait(timeout=2.0)
        return {
            "final_response": "Slow agent done",
            "risk_assessment": {"is_safe": True, "risk_level": "LOW"},
            "user_context": {"location": "Delhi"},
            "weather_data": {},
            "matched_sops": [],
            "retrieved_sops": [],
        }

    with patch("src.api.run_agent_turn", side_effect=slow_agent_turn):
        # Start slow request in a background thread
        def background_request():
            client.post("/api/v1/query", json={"query": "Slow query", "thread_id": "slow-session"})

        bg_thread = threading.Thread(target=background_request)
        bg_thread.start()

        # Wait until the slow agent is actively running
        assert agent_started.wait(timeout=1.0) is True

        # While agent is still running, verify locks are completely free
        assert not _rate_limit_lock.locked(), "Rate limit lock must not be held during agent execution"
        assert not _auth_lock.locked(), "Auth lock must not be held during agent execution"

        # Another thread can check rate limit and authenticate immediately (< 20ms)
        start_t = time.perf_counter()
        allowed, wait = check_rate_limit("independent-session", record=True)
        auth_ok = authenticate_session("independent-session", "")
        duration = time.perf_counter() - start_t

        assert allowed is True
        assert auth_ok is True
        assert duration < 0.1, f"Lock acquisition during slow operation took too long: {duration:.3f}s"

        # Release slow agent and join thread
        release_agent.set()
        bg_thread.join()


def test_sequential_rate_limit_thresholds_and_responses_unchanged(monkeypatch):
    """
    Test 5: Sequential calls verify rate-limit thresholds and backward-compatible manual recording.
    """
    monkeypatch.setattr(auth, "RATE_LIMIT_PER_MINUTE", 3)
    session_id = "test-thresholds-session"

    # 1. Atomic check-and-record
    assert check_rate_limit(session_id, record=True) == (True, 0)
    assert check_rate_limit(session_id, record=True) == (True, 0)
    assert check_rate_limit(session_id, record=True) == (True, 0)

    # 4th request rejected
    allowed, wait = check_rate_limit(session_id, record=True)
    assert allowed is False
    assert wait > 0

    # 2. Backward compatibility: record=False does not consume slot until record_query()
    session_b = "test-backward-compat-session"
    allowed, wait = check_rate_limit(session_b, record=False)
    assert allowed is True

    # Check multiple times without recording: should still be allowed
    for _ in range(5):
        allowed, _ = check_rate_limit(session_b, record=False)
        assert allowed is True

    # Now record queries manually
    record_query(session_b)
    record_query(session_b)
    record_query(session_b)

    # Now 4th check should fail
    allowed, wait = check_rate_limit(session_b, record=False)
    assert allowed is False
    assert wait > 0


def test_stateless_api_key_header_concurrency(monkeypatch):
    """
    Test 6: Stateless X-API-Key header verification under concurrent requests.
    """
    secret = "header-concurrency-token-999"
    monkeypatch.setattr(auth, "ACCESS_KEY", secret)
    client = TestClient(app)

    num_threads = 10
    barrier = threading.Barrier(num_threads)

    def send_auth_header(idx: int):
        barrier.wait()
        if idx % 2 == 0:
            return client.get("/api/v1/sops", headers={"X-API-Key": secret}).status_code
        else:
            return client.get("/api/v1/sops", headers={"X-API-Key": "wrong-token"}).status_code

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(send_auth_header, i) for i in range(num_threads)]
        statuses = [f.result() for f in futures]

    assert statuses.count(200) == 5
    assert statuses.count(401) == 5
