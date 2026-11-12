import json
import uuid
import pytest
from unittest.mock import MagicMock, patch
import requests

from frontend.streaming import parse_sse_stream, stream_consultation_events
from src.graph import run_agent_turn


class TestSSEParser:
    """Tests W3C SSE standard compliance in parse_sse_stream."""

    def test_parse_standard_events(self):
        lines = [
            "event: status",
            'data: {"stage": "input_guardrail", "message": "Validating..."}',
            "",
            "event: result",
            'data: {"advisory": "Take shade", "is_verified": true}',
            "",
            "event: done",
            'data: {"status": "completed"}',
            "",
        ]
        events = list(parse_sse_stream(lines))
        assert len(events) == 3
        assert events[0]["event"] == "status"
        assert events[0]["data"]["stage"] == "input_guardrail"
        assert events[1]["event"] == "result"
        assert events[1]["data"]["is_verified"] is True
        assert events[2]["event"] == "done"

    def test_parse_multiline_data(self):
        lines = [
            "event: message",
            "data: Line one of advisory",
            "data: Line two of advisory",
            "",
        ]
        events = list(parse_sse_stream(lines))
        assert len(events) == 1
        assert events[0]["event"] == "message"
        assert events[0]["data"]["raw"] == "Line one of advisory\nLine two of advisory"

    def test_parse_bytes_stream(self):
        lines = [
            b"event: status\r\n",
            b'data: {"stage": "fetch_weather"}\r\n',
            b"\r\n",
        ]
        events = list(parse_sse_stream(lines))
        assert len(events) == 1
        assert events[0]["event"] == "status"
        assert events[0]["data"]["stage"] == "fetch_weather"

    def test_ignore_sse_comments_and_pings(self):
        lines = [
            ": this is a comment",
            ": ping",
            "event: status",
            'data: {"stage": "ready"}',
            "",
        ]
        events = list(parse_sse_stream(lines))
        assert len(events) == 1
        assert events[0]["event"] == "status"
        assert events[0]["data"]["stage"] == "ready"

    def test_trailing_event_without_trailing_newline(self):
        lines = [
            "event: done",
            'data: {"status": "finished"}',
        ]
        events = list(parse_sse_stream(lines))
        assert len(events) == 1
        assert events[0]["event"] == "done"
        assert events[0]["data"]["status"] == "finished"

    def test_malformed_json_fallback(self):
        lines = [
            "event: custom",
            "data: not a valid json object",
            "",
        ]
        events = list(parse_sse_stream(lines))
        assert len(events) == 1
        assert events[0]["event"] == "custom"
        assert events[0]["data"] == {"raw": "not a valid json object"}


class TestFrontendStreamingClient:
    """Tests stream_consultation_events in HTTP and direct modes."""

    def test_direct_mode_successful_turn(self):
        tid = f"direct-test-{uuid.uuid4().hex[:6]}"
        events = list(stream_consultation_events(
            query="Can I jog in Bangalore this morning?",
            session_id=tid,
            api_url=None,  # Direct mode
        ))
        
        event_types = [e["event"] for e in events]
        assert "status" in event_types
        assert "result" in event_types
        assert "done" in event_types

        # Verify stages
        statuses = [e["data"]["stage"] for e in events if e["event"] == "status"]
        assert "input_guardrail" in statuses
        assert "parse_and_extract" in statuses
        assert "fetch_weather" in statuses

        # Verify verified advisory
        result_events = [e for e in events if e["event"] == "result"]
        assert len(result_events) == 1
        res = result_events[0]["data"]
        assert res["is_verified"] is True
        assert res["advisory"] is not None
        assert res["weather"] is not None

    def test_direct_mode_clarification(self):
        tid = f"direct-clarif-{uuid.uuid4().hex[:6]}"
        events = list(stream_consultation_events(
            query="Can I go for a jog?",
            session_id=tid,
            api_url=None,
        ))
        result_events = [e for e in events if e["event"] == "result"]
        assert len(result_events) == 1
        res = result_events[0]["data"]
        assert res["requires_clarification"] is True

    @patch("requests.post")
    def test_http_mode_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = [
            b"event: status",
            b'data: {"stage": "input_guardrail", "message": "Checking..."}',
            b"",
            b"event: result",
            b'data: {"advisory": "Approved outdoor plan", "is_verified": true}',
            b"",
            b"event: done",
            b'data: {"status": "completed"}',
            b"",
        ]
        mock_post.return_value = mock_resp

        events = list(stream_consultation_events(
            query="Is it safe in Pune?",
            session_id="http-test-1",
            api_url="http://localhost:8000",
            api_key="secret123",
        ))

        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "http://localhost:8000/api/v1/query/stream"
        assert kwargs["headers"]["X-API-Key"] == "secret123"

        assert len(events) == 3
        assert events[0]["event"] == "status"
        assert events[1]["event"] == "result"
        assert events[1]["data"]["is_verified"] is True

    @patch("requests.post")
    def test_http_mode_401_auth_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_post.return_value = mock_resp

        events = list(stream_consultation_events(
            query="Query",
            session_id="sid",
            api_url="http://localhost:8000",
            api_key="wrong_key",
        ))
        assert len(events) == 1
        assert events[0]["event"] == "error"
        assert "Authentication failed" in events[0]["data"]["error"]

    @patch("requests.post")
    def test_http_mode_429_rate_limit(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_post.return_value = mock_resp

        events = list(stream_consultation_events(
            query="Query",
            session_id="sid",
            api_url="http://localhost:8000",
        ))
        assert len(events) == 1
        assert events[0]["event"] == "error"
        assert "Rate limit exceeded" in events[0]["data"]["error"]

    @patch("requests.post")
    def test_http_mode_timeout_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")

        events = list(stream_consultation_events(
            query="Query",
            session_id="sid",
            api_url="http://localhost:8000",
        ))
        assert len(events) == 1
        assert events[0]["event"] == "error"
        assert "timed out" in events[0]["data"]["error"].lower()

    @patch("requests.post")
    def test_http_mode_connection_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError("Failed to connect")

        events = list(stream_consultation_events(
            query="Query",
            session_id="sid",
            api_url="http://localhost:8000",
        ))
        assert len(events) == 1
        assert events[0]["event"] == "error"
        assert "Unable to reach" in events[0]["data"]["error"]

    def test_streaming_conversation_persistence(self):
        """Verifies that direct streaming persists the consultation turn to SQLite."""
        from src.db import get_session_messages
        tid = f"persist-test-{uuid.uuid4().hex[:6]}"
        events = list(stream_consultation_events(
            query="Can I walk in Mumbai this evening?",
            session_id=tid,
            api_url=None,
        ))
        result_events = [e for e in events if e["event"] == "result"]
        assert len(result_events) == 1
        
        # Verify persistence
        msgs = get_session_messages(tid)
        assert len(msgs) >= 2
        user_msg = next((m for m in msgs if m["role"] == "user"), None)
        asst_msg = next((m for m in msgs if m["role"] == "assistant"), None)
        assert user_msg is not None
        assert asst_msg is not None
        assert "Mumbai" in user_msg["content"]
        assert asst_msg["content"] == result_events[0]["data"]["final_response"]

    def test_run_agent_turn_backward_compatibility(self):
        """Ensures that the existing synchronous run_agent_turn remains fully operational."""
        tid = f"compat-test-{uuid.uuid4().hex[:6]}"
        result = run_agent_turn(
            user_query="Can I exercise outside in Chennai?",
            session_id=tid,
        )
        assert isinstance(result, dict)
        assert "final_response" in result
        assert result["final_response"] is not None
        assert "route" in result

