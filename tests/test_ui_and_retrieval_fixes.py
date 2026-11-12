import pytest
import uuid
import time
from unittest.mock import MagicMock, patch
from src.db import save_turn, get_session_messages, create_or_update_session
from src.tools.sop_tool import SOPRetrieverTool
from src.schemas.context import UserContext
from src.nodes import evaluate_sops_node
from src.state import AgentState
from src.api import QueryResponse, app
from fastapi.testclient import TestClient


def test_sqlite_persistence_of_retrieved_sops():
    """Verify that retrieved SOP candidates and retrieval_error are saved and restored accurately."""
    session_id = f"test-persist-{uuid.uuid4().hex[:8]}"
    create_or_update_session(session_id, title="Persistence Test")

    sample_retrieved = [
        {"id": "SOP-LEISURE-001", "title": "Picnic Policy", "similarity_score": 0.892, "severity": "LOW"},
        {"id": "SOP-HEAT-001", "title": "Heat Wave Policy", "similarity_score": 0.741, "severity": "HIGH"},
    ]
    sample_active = {"id": "SOP-LEISURE-001", "title": "Picnic Policy", "severity": "LOW"}
    sample_matched = [sample_active]

    # Save turn with retrieved_sops
    save_turn(
        session_id=session_id,
        user_query="Can we plan a picnic in Kolkata?",
        assistant_response="Yes, conditions are safe.",
        route="direct_evaluation",
        citation="SOP-LEISURE-001",
        weather_data={"temperature_2m": 28.5, "precipitation": 0.0},
        trace=["Retrieved 2 SOPs"],
        matched_sops=sample_matched,
        active_sop=sample_active,
        retrieved_sops=sample_retrieved,
        retrieval_error=None,
    )

    # Retrieve and verify
    messages = get_session_messages(session_id)
    assert len(messages) == 2  # user and assistant
    assistant_msg = next(m for m in messages if m["role"] == "assistant")

    assert assistant_msg["retrieved_sops"] == sample_retrieved
    assert assistant_msg["active_sop"] == sample_active
    assert assistant_msg["matched_sops"] == sample_matched
    assert assistant_msg["retrieval_error"] is None
    assert assistant_msg["sop_citation"] == "SOP-LEISURE-001"


def test_sqlite_persistence_of_retrieval_error():
    """Verify that retrieval errors are persisted so UI can distinguish error from genuine empty."""
    session_id = f"test-error-{uuid.uuid4().hex[:8]}"
    create_or_update_session(session_id, title="Error Persist Test")

    save_turn(
        session_id=session_id,
        user_query="Test query with vector store error",
        assistant_response="Fallback advisory",
        route="deterministic_fallback",
        citation="SOP-FALLBACK",
        weather_data={"temperature_2m": 30.0},
        retrieved_sops=[],
        retrieval_error="ChromaDB connection refused",
    )

    messages = get_session_messages(session_id)
    assistant_msg = next(m for m in messages if m["role"] == "assistant")
    assert assistant_msg["retrieved_sops"] == []
    assert assistant_msg["retrieval_error"] == "ChromaDB connection refused"


def test_sop_retriever_tool_handles_exceptions():
    """Verify SOPRetrieverTool catches unexpected exceptions and returns error info rather than crashing."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.side_effect = RuntimeError("Embedding service timeout")

    tool = SOPRetrieverTool(retriever=mock_retriever)
    ctx = UserContext(activity="picnic", location="Kolkata")
    res = tool.run(query="picnic in Kolkata", context=ctx)

    assert res["success"] is False
    assert res["retrieved_sops"] == []
    assert res["count"] == 0
    assert "Embedding service timeout" in res["error"]


def test_evaluate_sops_node_distinguishes_governing_from_candidates():
    """Verify evaluate_sops_node populates retrieved_sops, matched_sops, and designates active_sop."""
    ctx = UserContext(activity="picnic", location="Kolkata", demographics=["all"])
    state: AgentState = {
        "session_id": "test-eval-1",
        "messages": [],
        "raw_query": "Can we plan an outdoor picnic in Kolkata this afternoon?",
        "user_context": ctx.model_dump(),
        "location": "Kolkata",
        "activity": "picnic",
        "timeframe": "this afternoon",
        "demographics": ["all"],
        "intent": "safety_inquiry",
        "pending_clarification": False,
        "clarification_target": None,
        "weather_data": {
            "temperature_2m": 29.9,
            "precipitation": 0.0,
            "precipitation_probability": 0,
            "wind_speed_10m": 8.0,
            "uv_index": 2.0,
            "location_name": "Kolkata",
        },
        "weather_error": None,
        "location_candidates": [],
        "retrieved_sops": [],
        "matched_sops": [],
        "active_sop": None,
        "retrieval_error": None,
        "risk_assessment": None,
        "sop_citation": None,
        "input_guardrail_passed": True,
        "grounding_valid": True,
        "grounding_errors": [],
        "guardrail_results": [],
        "retry_count": 0,
        "final_response": None,
        "route": None,
        "execution_trace": [],
    }

    result = evaluate_sops_node(state)
    assert "retrieved_sops" in result
    assert len(result["retrieved_sops"]) > 0

    # Verify candidate SOP structure
    for sop in result["retrieved_sops"]:
        assert "id" in sop
        assert "title" in sop
        assert "similarity_score" in sop

    # If matched, active_sop must be present and citation aligned
    if result.get("matched_sops"):
        assert result.get("active_sop") is not None
        assert result["active_sop"]["id"] in result["sop_citation"]
        # Governing SOP must be in the retrieved candidates
        retrieved_ids = [s["id"] for s in result["retrieved_sops"]]
        assert result["active_sop"]["id"] in retrieved_ids


def test_api_query_response_includes_retrieved_sops():
    """Verify FastAPI /api/v1/query returns retrieved_sops in response payload."""
    client = TestClient(app)
    response = client.post("/api/v1/query", json={"query": "Is it safe to cycle in Bhopal today?"})
    assert response.status_code == 200
    data = response.json()
    assert "retrieved_sops" in data
    assert isinstance(data["retrieved_sops"], list)
    assert "retrieval_error" in data


def test_copy_operation_preserves_session_integrity():
    """Verify that copying advisory content is non-destructive to session state and caches."""
    session_id = f"test-copy-{uuid.uuid4().hex[:8]}"
    create_or_update_session(session_id, title="Copy Test")
    
    advisory_text = "### Environmental Safety Advisory\nAll conditions normal in Kolkata."
    save_turn(
        session_id=session_id,
        user_query="Can we go to Kolkata?",
        assistant_response=advisory_text,
        route="direct_evaluation",
        citation="SOP-LEISURE-001",
        weather_data={"temperature_2m": 29.0},
    )

    # Verify conversation history before copy
    msgs_before = get_session_messages(session_id)
    assert len(msgs_before) == 2
    assert msgs_before[-1]["content"] == advisory_text

    # Simulated client-side copy reads the content attribute without touching DB or session state
    copied_text = msgs_before[-1]["content"]
    assert copied_text == advisory_text

    # Verify conversation history remains 100% intact after copy
    msgs_after = get_session_messages(session_id)
    assert len(msgs_after) == 2
    assert msgs_after[-1]["content"] == advisory_text
    assert msgs_after[-1]["sop_citation"] == "SOP-LEISURE-001"
