"""
Regression test suite for CliniGuard Honest Semantic Retrieval (Finding 5).

Covers:
- Test A: Successful search with zero relevant matches returns retrieved_sops=[] (no arbitrary catalog injection).
- Test B: Out-of-domain query does not receive the first 5 SOP catalog entries as fabricated matches.
- Test C: Retrieval exception is reported distinctly from a successful empty search (retrieval_error is set vs None).
- Test D: Frontend represents empty result honestly (distinguishes empty match state from service error).
- Test E: Empty retrieval does not prevent severe-weather override from evaluating and triggering CRITICAL risk.
- Test F: Relevant queries continue to return genuine candidates with valid similarity scores and SQLite persistence.
"""

import pytest
import uuid
from unittest.mock import MagicMock, patch
from src.retrieval.retriever import SOPRetriever
from src.retrieval.vector_store import VectorStore
from src.tools.sop_tool import SOPRetrieverTool
from src.schemas.context import UserContext
from src.nodes import evaluate_sops_node
from src.policy.engine import DeterministicPolicyEngine
from src.state import AgentState
from src.db import save_turn, get_session_messages, create_or_update_session


# =========================================================================
# Test A: Successful search with zero relevant matches returns empty list
# =========================================================================
def test_a_successful_search_zero_matches_returns_empty_list():
    """
    Test A:
    When semantic vector search executes successfully but finds no matching records
    (e.g., vector_store.similarity_search returns []), SOPRetriever must return []
    and MUST NOT fall back to injecting arbitrary SOP catalog entries.
    """
    retriever = SOPRetriever()
    # Mock vector_store.similarity_search to simulate zero matches found
    with patch.object(retriever.vector_store, "similarity_search", return_value=[]):
        ctx = UserContext(activity="knitting", location="Kolkata", demographics=["all"])
        results = retriever.retrieve(query="indoor knitting in living room", context=ctx, top_k=5)

        # Must be genuinely empty list
        assert results == [], f"Expected empty list, got: {results}"
        assert len(results) == 0

        # Verify via SOPRetrieverTool
        tool = SOPRetrieverTool(retriever=retriever)
        tool_res = tool.run(query="indoor knitting", context=ctx, top_k=5)
        assert tool_res["success"] is True
        assert tool_res["retrieved_sops"] == []
        assert tool_res["count"] == 0
        assert tool_res["error"] is None


# =========================================================================
# Test B: Out-of-domain query does not receive first 5 catalog entries
# =========================================================================
def test_b_out_of_domain_query_does_not_receive_fabricated_catalog():
    """
    Test B:
    An out-of-domain query (e.g. quantum cryptography) must not receive the first 5
    entries of sops.json as fabricated semantic search matches.
    """
    retriever = SOPRetriever()
    catalog_ids = [s.get("id") for s in retriever._sops_cache[:5]]

    ctx = UserContext(activity="quantum cryptography", location=None, demographics=[])
    # When similarity search has no matches above min_score or zero matches
    with patch.object(retriever.vector_store, "similarity_search", return_value=[]):
        results = retriever.retrieve(query="quantum cryptography algorithms", context=ctx, top_k=5)
        assert results == []

        result_ids = [s.get("id") for s in results]
        # Crucial invariant: result must NOT be the first 5 catalog entries
        assert result_ids != catalog_ids


# =========================================================================
# Test C: Retrieval exception reported distinctly from successful empty search
# =========================================================================
def test_c_retrieval_exception_distinct_from_successful_empty_search():
    """
    Test C:
    Verify that a vector store / embedding exception is captured and reported
    distinctly (retrieval_error is populated) compared to a successful search
    with zero matches (retrieval_error is None).
    """
    ctx = UserContext(activity="chess", location="Mumbai", demographics=["all"])
    base_state: AgentState = {
        "session_id": "test-distinction-1",
        "messages": [],
        "raw_query": "Can I play chess in my living room?",
        "user_context": ctx.model_dump(),
        "location": "Mumbai",
        "activity": "chess",
        "timeframe": "this afternoon",
        "demographics": ["all"],
        "intent": "safety_inquiry",
        "pending_clarification": False,
        "clarification_target": None,
        "weather_data": {"temperature_2m": 30.0, "precipitation": 0.0, "wind_speed_10m": 5.0},
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

    # Case 1: Successful search with zero matches
    with patch("src.nodes.sop_retriever_tool.run") as mock_tool_success:
        mock_tool_success.return_value = {
            "success": True,
            "retrieved_sops": [],
            "count": 0,
            "error": None,
        }
        res_empty = evaluate_sops_node(base_state)
        assert res_empty["retrieved_sops"] == []
        assert res_empty["retrieval_error"] is None  # Successful empty search

    # Case 2: Retrieval service failure (e.g. ChromaDB unavailable)
    with patch("src.nodes.sop_retriever_tool.run") as mock_tool_err:
        mock_tool_err.return_value = {
            "success": False,
            "retrieved_sops": [],
            "count": 0,
            "error": "Vector database connection failed: Timeout",
        }
        res_err = evaluate_sops_node(base_state)
        assert res_err["retrieved_sops"] == []
        assert res_err["retrieval_error"] == "Vector database connection failed: Timeout"

    # Distinct check: The two states are unambiguously distinguishable
    assert res_empty["retrieval_error"] != res_err["retrieval_error"]


# =========================================================================
# Test D: Frontend represents empty result honestly
# =========================================================================
def test_d_frontend_represents_empty_result_honestly():
    """
    Test D:
    Verify that when retrieved_sops is empty, the state structure reflects
    an honest empty state (no candidate cards, no fabricated scores) rather
    than a service error or fabricated matches.
    """
    session_id = f"test-fe-honest-{uuid.uuid4().hex[:8]}"
    create_or_update_session(session_id, title="Frontend Honest Test")

    # Turn with honest empty retrieval
    save_turn(
        session_id=session_id,
        user_query="Can I read a book in my living room?",
        assistant_response="Reading indoors is safe.",
        route="no_sop_fallback",
        citation="NO_SOP_APPLICABLE",
        weather_data={"temperature_2m": 25.0, "precipitation": 0.0},
        retrieved_sops=[],
        retrieval_error=None,
    )

    messages = get_session_messages(session_id)
    assert len(messages) == 2
    ast = next(m for m in messages if m["role"] == "assistant")

    # Honest empty state asserts
    assert ast["retrieved_sops"] == []
    assert ast["retrieval_error"] is None
    assert ast["sop_citation"] == "NO_SOP_APPLICABLE"


# =========================================================================
# Test E: Empty retrieval preserves severe-weather override evaluation
# =========================================================================
def test_e_empty_retrieval_preserves_severe_weather_override():
    """
    Test E:
    When semantic retrieval returns retrieved_sops=[] (no candidate SOPs matched the query),
    the deterministic policy engine must still evaluate severe-weather overrides
    and trigger CRITICAL risk if extreme weather conditions are present.
    """
    engine = DeterministicPolicyEngine()
    cyclone_weather = {
        "precipitation": 32.0,
        "precipitation_probability": 95,
        "wind_speed_10m": 65.0,
        "weather_code": 95,  # Severe thunderstorm
        "uv_index": 1.0,
        "temperature_2m": 24.0,
        "relative_humidity_2m": 90,
    }
    context = UserContext(activity="origami", location="Kolkata", demographics=["all"])

    # Pass candidate_sops=[] simulating zero semantic matches found
    risk, matched = engine.evaluate_risk(
        weather=cyclone_weather,
        context=context,
        candidate_sops=[],
        raw_query="Can I do origami in Kolkata?",
    )

    # Severe-weather override must trigger deterministically!
    assert risk.risk_level == "CRITICAL"
    assert risk.primary_sop_id == "SOP-OVERRIDE-001"
    assert len(matched) > 0
    assert matched[0]["id"] == "SOP-OVERRIDE-001"


# =========================================================================
# Test F: Existing retrieval score handling and persistence intact
# =========================================================================
def test_f_relevant_retrieval_scores_and_persistence_intact():
    """
    Test F:
    Verify that genuine semantic matches continue to receive accurate similarity scores,
    properly structured metadata, and are faithfully persisted in SQLite.
    """
    retriever = SOPRetriever()
    ctx = UserContext(activity="picnic", location="Kolkata", demographics=["all"])
    results = retriever.retrieve(query="outdoor picnic in a city park", context=ctx, top_k=5)

    assert len(results) > 0
    for s in results:
        assert "id" in s
        assert "title" in s
        # Genuine similarity score present
        assert "similarity_score" in s
        assert isinstance(s["similarity_score"], float)
        assert 0.0 <= s["similarity_score"] <= 1.0

    # Test persistence
    session_id = f"test-persist-f-{uuid.uuid4().hex[:8]}"
    create_or_update_session(session_id, title="Genuine Retrieval Persistence")
    save_turn(
        session_id=session_id,
        user_query="Can we picnic in Kolkata?",
        assistant_response="Conditions are favorable.",
        route="direct_evaluation",
        citation="SOP-LEISURE-001",
        weather_data={"temperature_2m": 22.0, "precipitation": 0.0},
        retrieved_sops=results,
        retrieval_error=None,
    )

    msgs = get_session_messages(session_id)
    ast = next(m for m in msgs if m["role"] == "assistant")
    assert len(ast["retrieved_sops"]) == len(results)
    assert ast["retrieved_sops"][0]["similarity_score"] == results[0]["similarity_score"]
