import os
import json
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from langgraph.checkpoint.sqlite import SqliteSaver

try:
    from src.config import DATABASE_PATH
except (ImportError, AttributeError):
    ROOT_DIR = Path(__file__).resolve().parent.parent
    DATABASE_PATH = os.getenv("DATABASE_PATH", str(ROOT_DIR / "data" / "medi_state.db"))
    if not os.path.isabs(DATABASE_PATH):
        DATABASE_PATH = str(ROOT_DIR / DATABASE_PATH)

# Ensure parent directory exists
os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)


def get_db_connection() -> sqlite3.Connection:
    """Returns a SQLite connection configured for concurrent access."""
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # Enable WAL mode for smooth multi-threading in Streamlit
    conn.execute("PRAGMA journal_mode=WAL;")
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


def get_sqlite_checkpointer() -> SqliteSaver:
    """Returns an initialized SqliteSaver checkpointer for LangGraph."""
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


def create_or_update_session(
    session_id: str,
    title: Optional[str] = None,
    location: Optional[str] = None,
    activity: Optional[str] = None,
    citation: Optional[str] = None,
):
    """Creates a new session or updates an existing session metadata."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db_connection()
    with conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()

        if not row:
            auto_title = title or f"Session {session_id[:8]}"
            conn.execute(
                """
                INSERT INTO sessions (session_id, title, created_at, updated_at, message_count, last_location, last_activity, last_sop_citation)
                VALUES (?, ?, ?, ?, 0, ?, ?, ?)
                """,
                (session_id, auto_title, now_str, now_str, location, activity, citation),
            )
        else:
            new_title = title or row["title"]
            new_loc = location or row["last_location"]
            new_act = activity or row["last_activity"]
            new_cit = citation or row["last_sop_citation"]
            conn.execute(
                """
                UPDATE sessions 
                SET title = ?, updated_at = ?, last_location = ?, last_activity = ?, last_sop_citation = ?
                WHERE session_id = ?
                """,
                (new_title, now_str, new_loc, new_act, new_cit, session_id),
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
):
    """Saves both user and assistant messages for a conversational turn."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Store weather data and associated SOP context
    payload_weather = dict(weather_data) if weather_data else {}
    if matched_sops:
        payload_weather["_matched_sops"] = matched_sops
    if active_sop:
        payload_weather["_active_sop"] = active_sop
        
    weather_str = json.dumps(payload_weather) if payload_weather else None
    trace_str = json.dumps(trace) if trace else None

    conn = get_db_connection()
    with conn:
        session_row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()

        if not session_row:
            if activity and location:
                title = f"{activity.title()} in {location.title()}"
            elif location:
                title = f"Weather in {location.title()}"
            elif activity:
                title = f"Safety for {activity.title()}"
            else:
                title = user_query[:35] + ("..." if len(user_query) > 35 else "")

            conn.execute(
                """
                INSERT INTO sessions (session_id, title, created_at, updated_at, message_count, last_location, last_activity, last_sop_citation)
                VALUES (?, ?, ?, ?, 0, ?, ?, ?)
                """,
                (session_id, title, now_str, now_str, location, activity, citation),
            )
        else:
            title = session_row["title"]
            if not title or title.startswith("Session") or title == "New Consultation":
                if activity and location:
                    title = f"{activity.title()} in {location.title()}"
                elif location:
                    title = f"Weather in {location.title()}"
                elif activity:
                    title = f"Safety for {activity.title()}"
                else:
                    title = user_query[:35] + ("..." if len(user_query) > 35 else "")

            new_loc = location or session_row["last_location"]
            new_act = activity or session_row["last_activity"]
            new_cit = citation or session_row["last_sop_citation"]
            conn.execute(
                """
                UPDATE sessions 
                SET title = ?, updated_at = ?, last_location = ?, last_activity = ?, last_sop_citation = ?
                WHERE session_id = ?
                """,
                (title, now_str, new_loc, new_act, new_cit, session_id),
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
        if weather_dict:
            matched_sops = weather_dict.pop("_matched_sops", [])
            active_sop = weather_dict.pop("_active_sop", None)
            
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
    """Deletes a session and all its associated messages."""
    conn = get_db_connection()
    with conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    conn.close()


def clear_all_sessions():
    """Clears all session and message history."""
    conn = get_db_connection()
    with conn:
        conn.execute("DELETE FROM messages")
        conn.execute("DELETE FROM sessions")
    conn.close()
