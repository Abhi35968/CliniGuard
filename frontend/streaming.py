"""
CliniGuard - Frontend Streaming Client & SSE Parser
===================================================

Provides Server-Sent Events (SSE) parsing and streaming client integration
for the Streamlit frontend. Supports dual-mode execution:
1. HTTP SSE Mode: Connects to FastAPI endpoint `POST /api/v1/query/stream`.
2. Direct Graph Mode: Consumes `stream_agent_turn` directly from LangGraph.

Complies with W3C SSE standard:
- Handles `event:` and `data:` fields.
- Handles multi-line data payloads joined with newlines.
- Handles empty lines as event delimiters.
- Ignores comments (lines starting with ':') and keep-alive pings.
- Safely parses JSON payloads with fallbacks.
- Gracefully handles network timeouts, auth errors, and rate limits.
"""

import json
import os
from typing import Any, Dict, Iterable, Iterator, Optional, Union
import requests

from src.graph import stream_agent_turn


def parse_sse_stream(line_iterator: Iterable[Union[str, bytes]]) -> Iterator[Dict[str, Any]]:
    """
    Parses a stream of Server-Sent Events (SSE) line-by-line according to W3C specs.

    Yields:
        Dict[str, Any]: {"event": event_type, "data": parsed_json_or_raw}
    """
    current_event = "message"
    data_buffer = []

    for raw_line in line_iterator:
        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8", errors="replace")
        else:
            line = raw_line

        line = line.rstrip("\r\n")

        # Comment or keep-alive ping
        if line.startswith(":"):
            continue

        # Blank line triggers event dispatch
        if not line:
            if data_buffer:
                raw_data = "\n".join(data_buffer)
                try:
                    payload = json.loads(raw_data)
                except Exception:
                    payload = {"raw": raw_data}
                yield {"event": current_event, "data": payload}
                data_buffer = []
                current_event = "message"
            continue

        # Field parsing
        if line.startswith("event:"):
            current_event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_content = line[len("data:"):]
            # According to SSE spec, if data: is followed by a single space, strip it
            if data_content.startswith(" "):
                data_content = data_content[1:]
            data_buffer.append(data_content)

    # Dispatch any remaining event in buffer if stream ends without trailing blank line
    if data_buffer:
        raw_data = "\n".join(data_buffer)
        try:
            payload = json.loads(raw_data)
        except Exception:
            payload = {"raw": raw_data}
        yield {"event": current_event, "data": payload}


def stream_consultation_events(
    query: str,
    session_id: str = "default-session",
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 60.0,
) -> Iterator[Dict[str, Any]]:
    """
    Streams consultation events for a user query.

    If `api_url` is provided, connects to the FastAPI SSE endpoint `POST /api/v1/query/stream`.
    Otherwise, invokes the LangGraph generator `stream_agent_turn` directly in-process.

    Yields structured events:
      - {"event": "status", "data": {"stage": str, "message": str}}
      - {"event": "provisional", "data": {"text": str, "is_verified": bool, "warning": str}}
      - {"event": "result", "data": dict (authoritative advisory & telemetry)}
      - {"event": "error", "data": {"error": str}}
      - {"event": "done", "data": {"status": "completed"}}
    """
    effective_api_url = (api_url or os.getenv("CLINIGUARD_API_URL", "")).strip().rstrip("/")

    # Mode 1: HTTP SSE via FastAPI
    if effective_api_url:
        endpoint = f"{effective_api_url}/api/v1/query/stream"
        effective_key = api_key or os.getenv("ACCESS_KEY", "").strip()
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        if effective_key:
            headers["X-API-Key"] = effective_key

        payload = {
            "query": query,
            "thread_id": session_id,
        }

        try:
            response = requests.post(
                endpoint,
                json=payload,
                headers=headers,
                stream=True,
                timeout=timeout,
            )

            # Handle HTTP errors safely without leaking credentials
            if response.status_code == 401:
                yield {
                    "event": "error",
                    "data": {"error": "Authentication failed: invalid or missing access key."},
                }
                return
            elif response.status_code == 429:
                yield {
                    "event": "error",
                    "data": {"error": "Rate limit exceeded. Please wait a moment before sending another query."},
                }
                return
            elif response.status_code != 200:
                yield {
                    "event": "error",
                    "data": {"error": f"Backend service responded with HTTP status {response.status_code}."},
                }
                return

            # Consume SSE stream
            for event in parse_sse_stream(response.iter_lines()):
                yield event

        except requests.exceptions.Timeout:
            yield {
                "event": "error",
                "data": {"error": "Request timed out while connecting to the CliniGuard streaming service."},
            }
        except requests.exceptions.ConnectionError:
            yield {
                "event": "error",
                "data": {"error": f"Unable to reach CliniGuard backend service at {effective_api_url}."},
            }
        except Exception as e:
            yield {
                "event": "error",
                "data": {"error": f"Streaming connection error: {str(e)}"},
            }
        return

    # Mode 2: Direct LangGraph In-Process Streaming
    try:
        for event in stream_agent_turn(user_query=query, session_id=session_id):
            yield event
    except Exception as e:
        yield {
            "event": "error",
            "data": {"error": f"Execution pipeline error: {str(e)}"},
        }
