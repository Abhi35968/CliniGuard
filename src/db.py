import os
import json
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except (ImportError, ModuleNotFoundError):
    try:
        from langgraph_checkpoint_sqlite import SqliteSaver
    except (ImportError, ModuleNotFoundError):
        SqliteSaver = None

from langgraph.checkpoint.memory import MemorySaver

try:
    from src.config import DATABASE_PATH
except (ImportError, AttributeError):
    ROOT_DIR = Path(__file__).resolve().parent.parent
    DATABASE_PATH = os.getenv("DATABASE_PATH", str(ROOT_DIR / "data" / "medi_state.db"))
    if not os.path.isabs(DATABASE_PATH):
        DATABASE_PATH = str(ROOT_DIR / DATABASE_PATH)

# Ensure parent directory exists
os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)

from src.observability import get_logger

logger = get_logger("db")



def get_db_connection(timeout: float = 30.0) -> sqlite3.Connection:
    """Returns a SQLite connection configured for concurrent access with WAL mode and a busy timeout."""
    conn = sqlite3.connect(DATABASE_PATH, timeout=timeout, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # Enable WAL mode, busy timeout, and normal synchronous mode for thread-safe concurrency
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(f"PRAGMA busy_timeout={int(timeout * 1000)};")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db():
    """Initializes the database schema for conversation and session storage."""
    conn = get_db_connection()
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT,
                created_at TEXT,
                updated_at TEXT,
                message_count INTEGER DEFAULT 0,
                last_location TEXT,
                last_activity TEXT,
                last_sop_citation TEXT
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp TEXT,
                route TEXT,
                sop_citation TEXT,
                weather_json TEXT,
                trace_json TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
            );
            """
        )
    conn.close()


# Initialize database schema on import
init_db()


def get_sqlite_checkpointer(timeout: float = 30.0):
    """Returns an initialized SqliteSaver checkpointer for LangGraph configured for concurrent WAL access."""
    if SqliteSaver is not None:
        try:
            conn = get_db_connection(timeout=timeout)
            saver = SqliteSaver(conn)
            if hasattr(saver, "setup"):
                saver.setup()
            return saver
        except Exception as e:
            logger.warning("Could not initialize SqliteSaver (%s). Using MemorySaver.", e)
    return MemorySaver()


def create_or_update_session(
    session_id: str,
    title: Optional[str] = None,
    location: Optional[str] = None,
    activity: Optional[str] = None,
    citation: Optional[str] = None,
):
    """Creates a new session or updates an existing session metadata atomically via UPSERT."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    auto_title = title or f"Session {session_id[:8]}"
    conn = get_db_connection()
    with conn:
        conn.execute(
            """
            INSERT INTO sessions (session_id, title, created_at, updated_at, message_count, last_location, last_activity, last_sop_citation)
            VALUES (?, ?, ?, ?, 0, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                title = COALESCE(excluded.title, sessions.title),
                updated_at = excluded.updated_at,
                last_location = COALESCE(excluded.last_location, sessions.last_location),
                last_activity = COALESCE(excluded.last_activity, sessions.last_activity),
                last_sop_citation = COALESCE(excluded.last_sop_citation, sessions.last_sop_citation);
            """,
            (session_id, auto_title, now_str, now_str, location, activity, citation),
        )
        if title:
            conn.execute(
                "UPDATE sessions SET title = ? WHERE session_id = ?",
                (title, session_id),
            )
    conn.close()


def save_turn(
    session_id: str,
    user_query: str,
    assistant_response: str,
    route: Optional[str] = None,
    citation: Optional[str] = None,
    weather_data: Optional[Dict[str, Any]] = None,
    trace: Optional[List[str]] = None,
    location: Optional[str] = None,
    activity: Optional[str] = None,
    matched_sops: Optional[List[Dict[str, Any]]] = None,
    active_sop: Optional[Dict[str, Any]] = None,
    retrieved_sops: Optional[List[Dict[str, Any]]] = None,
    retrieval_error: Optional[str] = None,
):
    """Saves both user and assistant messages for a conversational turn."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Store weather data and associated SOP context
    payload_weather = dict(weather_data) if weather_data else {}
    if matched_sops:
        payload_weather["_matched_sops"] = matched_sops
    if active_sop:
        payload_weather["_active_sop"] = active_sop
    if retrieved_sops is not None:
        payload_weather["_retrieved_sops"] = retrieved_sops
    if retrieval_error:
        payload_weather["_retrieval_error"] = retrieval_error
        
    weather_str = json.dumps(payload_weather) if payload_weather else None
    trace_str = json.dumps(trace) if trace else None

    conn = get_db_connection()
    with conn:
        # Determine descriptive title
        if activity and location:
            computed_title = f"{activity.title()} in {location.title()}"
        elif location:
            computed_title = f"Weather in {location.title()}"
        elif activity:
            computed_title = f"Safety for {activity.title()}"
        else:
            computed_title = user_query[:35] + ("..." if len(user_query) > 35 else "")

        # Atomic UPSERT ensuring session exists without race condition collisions
        conn.execute(
            """
            INSERT INTO sessions (session_id, title, created_at, updated_at, message_count, last_location, last_activity, last_sop_citation)
            VALUES (?, ?, ?, ?, 0, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                updated_at = excluded.updated_at,
                last_location = COALESCE(excluded.last_location, sessions.last_location),
                last_activity = COALESCE(excluded.last_activity, sessions.last_activity),
                last_sop_citation = COALESCE(excluded.last_sop_citation, sessions.last_sop_citation);
            """,
            (session_id, computed_title, now_str, now_str, location, activity, citation),
        )

        # Update title if current title is generic and a better descriptive title is available
        if computed_title:
            conn.execute(
                """
                UPDATE sessions
                SET title = ?
                WHERE session_id = ?
                  AND (title IS NULL OR title = '' OR title LIKE 'Session %' OR title = 'New Consultation')
                """,
                (computed_title, session_id),
            )

        # Insert User Message
        conn.execute(
            """
            INSERT INTO messages (session_id, role, content, timestamp, route, sop_citation, weather_json, trace_json)
            VALUES (?, 'user', ?, ?, ?, ?, ?, ?)
            """,
            (session_id, user_query, now_str, route, citation, weather_str, trace_str),
        )

        # Insert Assistant Message
        conn.execute(
            """
            INSERT INTO messages (session_id, role, content, timestamp, route, sop_citation, weather_json, trace_json)
            VALUES (?, 'assistant', ?, ?, ?, ?, ?, ?)
            """,
            (session_id, assistant_response, now_str, route, citation, weather_str, trace_str),
        )

        # Increment message count
        conn.execute(
            """
            UPDATE sessions 
            SET message_count = message_count + 2, updated_at = ?
            WHERE session_id = ?
            """,
            (now_str, session_id),
        )
    conn.close()


def get_session_messages(session_id: str) -> List[Dict[str, Any]]:
    """Retrieves all message records for a specific session."""
    conn = get_db_connection()
    rows = conn.execute(
        """
        SELECT role, content, timestamp, route, sop_citation, weather_json, trace_json
        FROM messages
        WHERE session_id = ?
        ORDER BY id ASC
        """,
        (session_id,),
    ).fetchall()
    conn.close()

    result = []
    for r in rows:
        weather_dict = json.loads(r["weather_json"]) if r["weather_json"] else None
        trace_list = json.loads(r["trace_json"]) if r["trace_json"] else []
        matched_sops = []
        active_sop = None
        retrieved_sops = []
        retrieval_error = None
        if weather_dict:
            matched_sops = weather_dict.pop("_matched_sops", [])
            active_sop = weather_dict.pop("_active_sop", None)
            retrieved_sops = weather_dict.pop("_retrieved_sops", [])
            retrieval_error = weather_dict.pop("_retrieval_error", None)
            
        result.append(
            {
                "role": r["role"],
                "content": r["content"],
                "timestamp": r["timestamp"],
                "route": r["route"],
                "sop_citation": r["sop_citation"],
                "weather_data": weather_dict,
                "matched_sops": matched_sops,
                "active_sop": active_sop,
                "retrieved_sops": retrieved_sops,
                "retrieval_error": retrieval_error,
                "execution_trace": trace_list,
            }
        )
    return result


def list_all_sessions(include_empty: bool = False) -> List[Dict[str, Any]]:
    """Lists all active sessions with metadata, sorted by most recently updated."""
    conn = get_db_connection()
    if include_empty:
        rows = conn.execute(
            """
            SELECT session_id, title, created_at, updated_at, message_count, last_location, last_activity, last_sop_citation
            FROM sessions
            ORDER BY updated_at DESC
            """
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT session_id, title, created_at, updated_at, message_count, last_location, last_activity, last_sop_citation
            FROM sessions
            WHERE message_count > 0
            ORDER BY updated_at DESC
            """
        ).fetchall()
    conn.close()

    return [dict(r) for r in rows]


def delete_session(session_id: str):
    """Deletes a session, all its associated messages, and its LangGraph checkpoints."""
    try:
        from src.graph import session_scope_lock
        ctx = session_scope_lock(session_id)
    except ImportError:
        from contextlib import nullcontext
        ctx = nullcontext()

    with ctx:
        conn = get_db_connection()
        with conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            try:
                conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (str(session_id),))
                conn.execute("DELETE FROM writes WHERE thread_id = ?", (str(session_id),))
            except Exception:
                pass
        conn.close()


def clear_all_sessions():
    """Clears all session, message history, and LangGraph checkpoints."""
    try:
        from src.graph import turn_coordinator, _graph_lock
        turn_ctx = turn_coordinator.exclusive()
        graph_ctx = _graph_lock
    except ImportError:
        from contextlib import nullcontext
        turn_ctx = nullcontext()
        graph_ctx = nullcontext()

    with turn_ctx:
        with graph_ctx:
            conn = get_db_connection()
            with conn:
                conn.execute("DELETE FROM messages")
                conn.execute("DELETE FROM sessions")
                try:
                    conn.execute("DELETE FROM checkpoints")
                    conn.execute("DELETE FROM writes")
                except Exception:
                    pass
            conn.close()

            # Safely clear in-memory session locks since zero turns are active
            try:
                from src.graph import _session_locks, _session_locks_guard
                with _session_locks_guard:
                    _session_locks.clear()
            except Exception:
                pass


def backup_database(destination_path: Optional[str] = None, timeout: float = 30.0) -> str:
    """
    Performs an online, consistent backup of the SQLite database using SQLite's native backup API.
    Compatible with WAL mode while readers and writers are concurrently active.
    Returns the absolute path to the created backup file.
    """
    if destination_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(os.path.dirname(DATABASE_PATH), "backups")
        os.makedirs(backup_dir, exist_ok=True)
        destination_path = os.path.join(backup_dir, f"medi_state_backup_{timestamp}.db")
    else:
        dest_dir = os.path.dirname(destination_path)
        if dest_dir:
            os.makedirs(dest_dir, exist_ok=True)

    source_conn = get_db_connection(timeout=timeout)
    dest_conn = sqlite3.connect(destination_path)
    try:
        with dest_conn:
            source_conn.backup(dest_conn)
        logger.info("Online database backup created at %s", destination_path)
        return os.path.abspath(destination_path)
    finally:
        dest_conn.close()
        source_conn.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "backup":
        dest = sys.argv[2] if len(sys.argv) > 2 else None
        target = backup_database(dest)
        print(f"Database backup completed: {target}")
    else:
        init_db()
        print(f"Database initialized at: {DATABASE_PATH}")
