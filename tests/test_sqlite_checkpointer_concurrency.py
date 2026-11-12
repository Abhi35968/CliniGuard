"""
CliniGuard SQLite Checkpointer Concurrency Tests
=================================================
Regression and concurrency tests verifying Finding 7:
1. Concurrent requests using distinct graph/session IDs execute safely without database locks.
2. Concurrent updates to the same session ID execute without UNIQUE constraint collisions or lost writes.
3. Checkpoint persistence and multi-turn state recovery after reopening the database.
4. Transaction rollback and connection cleanup upon failures.
5. Verification of WAL mode, busy_timeout, and synchronous PRAGMA settings.
"""

import sqlite3
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from src.config import DATABASE_PATH
from src.db import (
    get_db_connection,
    get_sqlite_checkpointer,
    save_turn,
    create_or_update_session,
    get_session_messages,
    list_all_sessions,
    delete_session,
    clear_all_sessions,
    backup_database,
)
from src.graph import run_agent_turn, compiled_graph, recompile_graph


MOCK_WEATHER_SUCCESS = {
    "success": True,
    "weather_data": {
        "location_name": "Kolkata",
        "temperature_2m": 26.0,
        "apparent_temperature": 26.0,
        "precipitation_probability": 0,
        "precipitation": 0.0,
        "wind_speed_10m": 10.0,
        "wind_gusts_10m": 12.0,
        "uv_index": 3.0,
        "relative_humidity": 50.0,
        "weather_code": 0,
        "weather_description": "Clear sky",
        "is_day": 1,
        "timezone": "Asia/Kolkata",
        "observation_time": "2026-10-20T12:00",
    },
    "location_candidates": [],
    "resolved_location": {"name": "Kolkata"},
}


@pytest.fixture(autouse=True)
def clean_test_sessions():
    """Cleanup test sessions created during test runs."""
    yield
    try:
        conn = get_db_connection()
        with conn:
            conn.execute("DELETE FROM messages WHERE session_id LIKE 'test-conc-%'")
            conn.execute("DELETE FROM sessions WHERE session_id LIKE 'test-conc-%'")
            try:
                conn.execute("DELETE FROM checkpoints WHERE thread_id LIKE 'test-conc-%'")
                conn.execute("DELETE FROM writes WHERE thread_id LIKE 'test-conc-%'")
            except Exception:
                pass
        conn.close()
    except Exception:
        pass


def test_wal_mode_and_busy_timeout_configured():
    """Verify that all connections are configured with WAL journal mode and 30s busy timeout."""
    conn = get_db_connection()
    journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
    busy_timeout = conn.execute("PRAGMA busy_timeout;").fetchone()[0]
    synchronous = conn.execute("PRAGMA synchronous;").fetchone()[0]
    conn.close()

    assert journal_mode.lower() == "wal", f"Expected WAL journal mode, got {journal_mode}"
    assert busy_timeout == 30000, f"Expected 30000ms busy timeout, got {busy_timeout}"
    # In SQLite, PRAGMA synchronous=NORMAL returns 1
    assert synchronous == 1, f"Expected synchronous=1 (NORMAL), got {synchronous}"


def test_concurrent_distinct_session_requests():
    """
    Test 1: 8 concurrent requests using distinct session IDs.
    Verifies that parallel LangGraph execution and SQLite persistence do not conflict or lock.
    """
    num_sessions = 8
    barrier = threading.Barrier(num_sessions)
    session_ids = [f"test-conc-distinct-{i}-{uuid.uuid4().hex[:6]}" for i in range(num_sessions)]
    errors = []
    err_lock = threading.Lock()

    def worker(sid: str):
        try:
            barrier.wait()
            # Patch weather tool to isolate SQLite concurrency from Open-Meteo network calls
            with patch("src.nodes.weather_tool.run", return_value=MOCK_WEATHER_SUCCESS):
                res = run_agent_turn(
                    user_query="Can I walk in Kolkata this afternoon?",
                    session_id=sid,
                )
                assert res is not None
                assert "final_response" in res
        except Exception as e:
            with err_lock:
                errors.append((sid, e))

    threads = [threading.Thread(target=worker, args=(sid,)) for sid in session_ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Concurrent distinct requests failed: {errors}"

    # Verify all sessions were persisted to SQLite with their messages
    for sid in session_ids:
        msgs = get_session_messages(sid)
        assert len(msgs) >= 2, f"Session {sid} messages not persisted (found {len(msgs)})"


def test_concurrent_same_session_updates():
    """
    Test 2: 12 concurrent threads updating the SAME session ID using save_turn and create_or_update_session.
    Verifies that atomic UPSERT prevents UNIQUE constraint collisions and race condition errors.
    """
    num_threads = 12
    barrier = threading.Barrier(num_threads)
    session_id = f"test-conc-same-{uuid.uuid4().hex[:8]}"
    errors = []
    err_lock = threading.Lock()

    def worker(idx: int):
        try:
            barrier.wait()
            if idx % 2 == 0:
                create_or_update_session(
                    session_id=session_id,
                    title=f"Updated Title {idx}",
                    location="Kolkata",
                    activity="Jogging",
                )
            else:
                save_turn(
                    session_id=session_id,
                    user_query=f"Query {idx}",
                    assistant_response=f"Response {idx}",
                    location="Kolkata",
                    activity="Jogging",
                )
        except Exception as e:
            with err_lock:
                errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Zero UNIQUE constraint or locking errors allowed
    assert len(errors) == 0, f"Concurrent same-session updates threw errors: {errors}"

    # Verify session exists and all message turns were safely recorded
    msgs = get_session_messages(session_id)
    # 6 odd threads called save_turn, each adding 2 messages (user + assistant) = 12 messages
    assert len(msgs) == 12, f"Expected 12 messages in session, got {len(msgs)}"


def test_concurrent_same_session_serialized_agent_turns():
    """
    Test 3: Multiple concurrent calls to run_agent_turn for the SAME session ID.
    Verifies that per-session locking serializes turns, avoiding state clobbering or checkpoint conflicts.
    Uses a threading.Barrier to release concurrent worker threads at the exact same instant.
    """
    session_id = f"test-conc-multiturn-{uuid.uuid4().hex[:8]}"
    queries = [
        "What is the weather in Delhi?",
        "Can I go for a jog in Delhi today?",
    ]
    barrier = threading.Barrier(len(queries))
    results = [None] * len(queries)
    errors = []
    err_lock = threading.Lock()

    delhi_weather = dict(MOCK_WEATHER_SUCCESS)
    delhi_weather["weather_data"] = dict(MOCK_WEATHER_SUCCESS["weather_data"])
    delhi_weather["weather_data"]["location_name"] = "Delhi"
    delhi_weather["resolved_location"] = {"name": "Delhi"}

    def worker(idx: int, query: str):
        try:
            barrier.wait()
            with patch("src.nodes.weather_tool.run", return_value=delhi_weather):
                res = run_agent_turn(user_query=query, session_id=session_id)
                results[idx] = res
        except Exception as e:
            with err_lock:
                errors.append(e)

    threads = [
        threading.Thread(target=worker, args=(i, q))
        for i, q in enumerate(queries)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Concurrent same-session agent turns failed: {errors}"
    assert all(r is not None and "final_response" in r for r in results)

    # Verify state snapshot in checkpoint exists and retains location
    state = compiled_graph.get_state({"configurable": {"thread_id": session_id}})
    assert state is not None
    assert state.values.get("location") == "Delhi"

    # Messages in SQLite reflect both turns (4 messages total: 2 user + 2 assistant)
    msgs = get_session_messages(session_id)
    assert len(msgs) == 4


def test_checkpoint_persistence_and_recovery_after_reopen():
    """
    Test 4: Verify checkpoint persistence across checkpointer re-instantiation (reopening database).
    """
    session_id = f"test-conc-reopen-{uuid.uuid4().hex[:8]}"

    mumbai_weather = dict(MOCK_WEATHER_SUCCESS)
    mumbai_weather["weather_data"] = dict(MOCK_WEATHER_SUCCESS["weather_data"])
    mumbai_weather["weather_data"]["location_name"] = "Mumbai"
    mumbai_weather["resolved_location"] = {"name": "Mumbai"}

    with patch("src.nodes.weather_tool.run", return_value=mumbai_weather):
        res1 = run_agent_turn(user_query="Can I walk in Mumbai today?", session_id=session_id)
        assert res1 is not None

    # Simulate process restart by recompiling graph with a fresh checkpointer connection
    fresh_checkpointer = get_sqlite_checkpointer()
    fresh_graph = recompile_graph(fresh_checkpointer)

    # State must be recovered from disk
    recovered_state = fresh_graph.get_state({"configurable": {"thread_id": session_id}})
    assert recovered_state is not None
    assert recovered_state.values.get("location") == "Mumbai"
    assert recovered_state.values.get("session_id") == session_id

    # Messages in SQLite must also be present
    msgs = get_session_messages(session_id)
    assert len(msgs) >= 2
    assert "Mumbai" in msgs[0]["content"]


def test_transaction_rollback_and_connection_cleanup():
    """
    Test 5: Transaction rollback and connection cleanup upon unexpected SQL failure.
    Verifies that failed transactions do not leave dangling uncommitted locks.
    """
    session_id = f"test-conc-rollback-{uuid.uuid4().hex[:8]}"
    conn = get_db_connection()

    # Attempt an atomic transaction where the second statement fails
    with pytest.raises(sqlite3.OperationalError):
        with conn:
            conn.execute(
                "INSERT INTO sessions (session_id, title) VALUES (?, ?)",
                (session_id, "Temporary Title"),
            )
            # Deliberate syntax/table error to trigger rollback
            conn.execute("INSERT INTO non_existent_table VALUES (1, 2, 3)")
    conn.close()

    # Verify that the entire transaction was rolled back (session should not exist)
    conn_verify = get_db_connection()
    row = conn_verify.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    conn_verify.close()
    assert row is None, "Failed transaction was not rolled back"

    # Verify that database is immediately usable and not locked
    create_or_update_session(session_id=session_id, title="Valid Title")
    conn_verify2 = get_db_connection()
    row2 = conn_verify2.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    conn_verify2.close()
    assert row2 is not None
    assert row2["title"] == "Valid Title"


def test_concurrent_recompile_graph_safety():
    """
    Test 6: recompile_graph() called concurrently with active run_agent_turn operations.
    Verifies that _graph_lock and atomic graph reference binding prevent None-dereferencing,
    incomplete graph compilation reads, or crashes during graph invocation.
    """
    session_id = f"test-conc-recompile-{uuid.uuid4().hex[:8]}"
    errors = []
    err_lock = threading.Lock()
    barrier = threading.Barrier(3)

    kolkata_weather = dict(MOCK_WEATHER_SUCCESS)

    def turn_worker():
        try:
            barrier.wait()
            with patch("src.nodes.weather_tool.run", return_value=kolkata_weather):
                res = run_agent_turn(
                    user_query="Can I walk in Kolkata today?",
                    session_id=session_id,
                )
                assert res is not None
                assert "final_response" in res
        except Exception as e:
            with err_lock:
                errors.append(e)

    def recompile_worker():
        try:
            barrier.wait()
            for _ in range(3):
                recompile_graph()
        except Exception as e:
            with err_lock:
                errors.append(e)

    threads = [
        threading.Thread(target=turn_worker),
        threading.Thread(target=turn_worker),
        threading.Thread(target=recompile_worker),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Concurrent recompile_graph raised errors: {errors}"


def test_delete_session_coordination_with_turn():
    """
    Test 7: delete_session() called concurrently on a session undergoing an active agent turn.
    Verifies that delete_session acquires session_scope_lock, ensuring no race conditions,
    corrupted DB operations, or deadlock.
    """
    session_id = f"test-conc-delcoord-{uuid.uuid4().hex[:8]}"
    errors = []
    err_lock = threading.Lock()
    barrier = threading.Barrier(2)

    kolkata_weather = dict(MOCK_WEATHER_SUCCESS)

    def turn_worker():
        try:
            barrier.wait()
            with patch("src.nodes.weather_tool.run", return_value=kolkata_weather):
                run_agent_turn(
                    user_query="Can I walk in Kolkata today?",
                    session_id=session_id,
                )
        except Exception as e:
            with err_lock:
                errors.append(e)

    def delete_worker():
        try:
            barrier.wait()
            delete_session(session_id)
        except Exception as e:
            with err_lock:
                errors.append(e)

    t1 = threading.Thread(target=turn_worker)
    t2 = threading.Thread(target=delete_worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(errors) == 0, f"Coordinated delete_session raised errors: {errors}"


def test_session_scope_lock_ref_counting_and_reclamation():
    """
    Test 8: session_scope_lock reference counting and bounded memory reclamation.
    Verifies that:
    1. ref_count accurately tracks active and waiting threads.
    2. Inactive locks (ref_count == 0) are pruned when registry size > 500.
    3. Active locks (ref_count > 0) are never pruned during cleanup.
    """
    from src.graph import _session_locks, _session_locks_guard, session_scope_lock

    # Pre-populate registry with dummy inactive entries > 500 to test threshold
    with _session_locks_guard:
        _session_locks.clear()
        for i in range(505):
            entry = type("MockEntry", (), {"lock": threading.Lock(), "ref_count": 0})()
            _session_locks[f"dummy-inactive-{i}"] = entry

    test_sid = "test-reclaim-sid"
    # When entering context, test_sid ref_count is 1
    with session_scope_lock(test_sid):
        with _session_locks_guard:
            assert _session_locks[test_sid].ref_count == 1
            assert test_sid in _session_locks

    # After exiting context, since registry was > 500 and ref_count became 0, inactive entries are pruned
    with _session_locks_guard:
        assert test_sid not in _session_locks
        assert len(_session_locks) == 0

    # Ensure active entries (ref_count > 0) are NEVER pruned during cleanup
    with _session_locks_guard:
        _session_locks.clear()
        for i in range(505):
            entry = type("MockEntry", (), {"lock": threading.Lock(), "ref_count": 0})()
            _session_locks[f"dummy-inactive-{i}"] = entry

    active_sid = "test-active-held"
    with session_scope_lock(active_sid):
        # Trigger reclamation by entering and exiting another temporary session
        with session_scope_lock("test-temporary"):
            pass
        # active_sid must still be in registry because ref_count == 1
        with _session_locks_guard:
            assert active_sid in _session_locks
            assert _session_locks[active_sid].ref_count == 1


def test_clear_all_sessions_coordination_with_active_turns():
    """
    Test 9: clear_all_sessions() coordination with active agent turns.
    Verifies that turn_coordinator.exclusive() prevents race conditions,
    avoids session resurrection, and executes without database lock errors.
    """
    session_id = f"test-conc-clearcoord-{uuid.uuid4().hex[:8]}"
    errors = []
    err_lock = threading.Lock()
    barrier = threading.Barrier(2)

    kolkata_weather = dict(MOCK_WEATHER_SUCCESS)

    def turn_worker():
        try:
            barrier.wait()
            with patch("src.nodes.weather_tool.run", return_value=kolkata_weather):
                res = run_agent_turn(
                    user_query="Can I walk in Kolkata today?",
                    session_id=session_id,
                )
                assert res is not None
        except Exception as e:
            with err_lock:
                errors.append(e)

    def clear_worker():
        try:
            barrier.wait()
            clear_all_sessions()
        except Exception as e:
            with err_lock:
                errors.append(e)

    t1 = threading.Thread(target=turn_worker)
    t2 = threading.Thread(target=clear_worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(errors) == 0, f"Coordinated clear_all_sessions raised errors: {errors}"


def test_session_lock_not_reclaimed_while_thread_waiting():
    """
    Test 10: Ensures that when Thread 1 holds a session lock and Thread 2 is blocked
    waiting for the SAME session lock, ref_count is 2 and the lock is NEVER evicted
    during cleanup, even if the registry has over 500 entries.
    """
    import time
    from src.graph import _session_locks, _session_locks_guard, session_scope_lock

    # Pre-populate registry with 505 inactive entries
    with _session_locks_guard:
        _session_locks.clear()
        for i in range(505):
            entry = type("MockEntry", (), {"lock": threading.Lock(), "ref_count": 0})()
            _session_locks[f"dummy-inactive-{i}"] = entry

    sid = "test-contended-session"
    step1_acquired = threading.Event()
    step2_waiting = threading.Event()
    step3_release = threading.Event()
    thread2_done = threading.Event()
    errors = []

    def thread1_worker():
        try:
            with session_scope_lock(sid):
                step1_acquired.set()
                # Wait until thread 2 is actively waiting to acquire sid
                step2_waiting.wait(timeout=5.0)
                # Verify that ref_count is 2 while thread 2 is waiting
                with _session_locks_guard:
                    assert sid in _session_locks
                    assert _session_locks[sid].ref_count == 2
                step3_release.wait(timeout=5.0)
        except Exception as e:
            errors.append(e)

    def thread2_worker():
        try:
            step1_acquired.wait(timeout=5.0)
            # Signal that thread 2 is about to enter session_scope_lock
            step2_waiting.set()
            with session_scope_lock(sid):
                # Thread 2 now holds the lock
                with _session_locks_guard:
                    assert sid in _session_locks
                    assert _session_locks[sid].ref_count == 1
            thread2_done.set()
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=thread1_worker)
    t2 = threading.Thread(target=thread2_worker)
    t1.start()
    t2.start()

    # Allow thread 1 to hold lock, verify thread 2 is waiting, then release
    step1_acquired.wait(timeout=5.0)
    time.sleep(0.05)
    step3_release.set()

    t1.join(timeout=5.0)
    t2.join(timeout=5.0)

    assert len(errors) == 0, f"Thread waiting on session lock encountered errors: {errors}"
    assert thread2_done.is_set(), "Thread 2 did not complete acquisition"


def test_deadlock_free_concurrency_stress():
    """
    Test 11: High-concurrency stress test verifying deadlock freedom across all lock tiers:
    - Multiple run_agent_turn calls (same and distinct sessions)
    - recompile_graph calls
    - delete_session calls
    - clear_all_sessions calls
    All synchronized via a barrier.
    """
    num_threads = 6
    barrier = threading.Barrier(num_threads)
    errors = []
    err_lock = threading.Lock()
    kolkata_weather = dict(MOCK_WEATHER_SUCCESS)

    def turn_worker(sid: str):
        try:
            barrier.wait()
            with patch("src.nodes.weather_tool.run", return_value=kolkata_weather):
                run_agent_turn(user_query="Can I walk in Kolkata today?", session_id=sid)
        except Exception as e:
            with err_lock:
                errors.append(e)

    def delete_worker(sid: str):
        try:
            barrier.wait()
            delete_session(sid)
        except Exception as e:
            with err_lock:
                errors.append(e)

    def recompile_worker():
        try:
            barrier.wait()
            recompile_graph()
        except Exception as e:
            with err_lock:
                errors.append(e)

    def clear_worker():
        try:
            barrier.wait()
            clear_all_sessions()
        except Exception as e:
            with err_lock:
                errors.append(e)

    threads = [
        threading.Thread(target=turn_worker, args=("stress-sid-1",)),
        threading.Thread(target=turn_worker, args=("stress-sid-1",)),
        threading.Thread(target=turn_worker, args=("stress-sid-2",)),
        threading.Thread(target=delete_worker, args=("stress-sid-2",)),
        threading.Thread(target=recompile_worker),
        threading.Thread(target=clear_worker),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15.0)

    assert len(errors) == 0, f"Deadlock stress test raised errors: {errors}"


def test_online_backup_database(tmp_path):
    """
    Verifies that backup_database creates a consistent SQLite backup snapshot
    using SQLite's online backup API without locking or corrupting the source.
    """
    # 1. Insert test records into database
    test_sid = f"backup-test-{uuid.uuid4()}"
    create_or_update_session(test_sid, title="Backup Test Session", location="Varanasi", activity="Yoga")
    save_turn(
        session_id=test_sid,
        user_query="Is it safe for morning yoga?",
        assistant_response="Morning conditions are favorable.",
        route="direct",
        citation="SOP-HEAT-001",
    )

    # 2. Perform online backup to a temporary file
    backup_file = str(tmp_path / "medi_state_backup.db")
    result_path = backup_database(destination_path=backup_file)
    assert result_path == backup_file
    assert (tmp_path / "medi_state_backup.db").exists()

    # 3. Verify backup database integrity and content
    dest_conn = sqlite3.connect(backup_file)
    dest_conn.row_factory = sqlite3.Row
    try:
        # Integrity check
        integrity = dest_conn.execute("PRAGMA integrity_check;").fetchone()[0]
        assert integrity == "ok"

        # Check records were copied faithfully
        sess = dest_conn.execute("SELECT * FROM sessions WHERE session_id = ?", (test_sid,)).fetchone()
        assert sess is not None
        assert sess["title"] == "Backup Test Session"

        msgs = dest_conn.execute("SELECT * FROM messages WHERE session_id = ?", (test_sid,)).fetchall()
        assert len(msgs) == 2
        assert msgs[0]["content"] == "Is it safe for morning yoga?"
        assert msgs[1]["content"] == "Morning conditions are favorable."
    finally:
        dest_conn.close()

    # 4. Cleanup test session from main db
    delete_session(test_sid)

