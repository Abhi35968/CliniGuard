import uuid
import pytest
from src.graph import run_agent_turn
from src.nodes import validate_grounding_node, check_grounding_condition, deterministic_fallback_node


def test_clarification_branch_when_location_missing():
    # User asks without mentioning any location
    sid = f"test-no-loc-{uuid.uuid4().hex[:8]}"
    result = run_agent_turn("Is it safe to cycle outside?", session_id=sid)
    assert result.get("route") == "clarification"
    assert "location" in result["final_response"].lower() or "city" in result["final_response"].lower()
    assert result.get("sop_citation") == "CLARIFICATION_REQUIRED"
    assert result.get("pending_clarification") is True


def test_mid_clarification_resolution_turn():
    # Turn 1: Ask without location -> triggers clarification
    sid = f"test-clarify-flow-{uuid.uuid4().hex[:8]}"
    res1 = run_agent_turn("Can I go for a jog outside?", session_id=sid)
    assert res1.get("route") == "clarification"
    assert res1.get("pending_clarification") is True
    assert res1.get("activity") in ["running", "jogging", "jog", "outdoor exercise"]

    # Turn 2: User answers directly with just the city name "In Delhi"
    res2 = run_agent_turn("In Delhi", session_id=sid)
    assert res2.get("location") == "Delhi"
    assert res2.get("activity") in ["running", "jogging", "jog", "outdoor exercise"]
    assert res2.get("pending_clarification") is False
    assert res2.get("route") in ["generate_advisory", "no_sop_fallback", "deterministic_fallback"]


def test_weather_error_branch_for_invalid_location():
    # User provides non-existent location
    sid = f"test-bad-loc-{uuid.uuid4().hex[:8]}"
    result = run_agent_turn("Can I hike in FakeGhostTownXYZ12345 today?", session_id=sid)
    assert result.get("route") == "weather_error"
    assert "unable to retrieve" in result["final_response"].lower()
    assert result.get("sop_citation") == "API_UNAVAILABLE_HONEST_FALLBACK"


def test_no_sop_branch_for_uncovered_activity():
    # User asks about indoor activity with no outdoor weather hazard
    sid = f"test-indoor-{uuid.uuid4().hex[:8]}"
    result = run_agent_turn("Can I play chess in my living room in Mumbai?", session_id=sid)
    assert result.get("route") == "no_sop_fallback"
    assert result.get("sop_citation") == "NO_SOP_APPLICABLE"
    assert "does not currently have a formal standard operating procedure" in result["final_response"].lower()


def test_multi_turn_session_continuity():
    # Turn 1: Establish location and activity
    sid = f"session-continuity-{uuid.uuid4().hex[:8]}"
    res1 = run_agent_turn("Is it safe to bike in Bhopal today?", session_id=sid)
    assert res1.get("location") == "Bhopal"
    assert res1.get("activity") in ["cycling", "biking", "bike", "two-wheeler"]

    # Turn 2: Follow-up question relying on previous turn context
    res2 = run_agent_turn("What about this evening instead?", session_id=sid)
    assert res2.get("location") == "Bhopal"
    assert res2.get("activity") in ["cycling", "biking", "bike", "two-wheeler"]
    assert "evening" in res2.get("timeframe", "").lower()


def test_guardrail_validation_failure_and_fallback():
    # State simulating an LLM output that failed to cite the required SOP ID
    invalid_state = {
        "final_response": "You are totally fine to bike outside today! Have a great ride.",
        "active_sop": {
            "id": "SOP-EXER-002",
            "title": "Squally Wind & Crosswind Hazard",
            "severity": "HIGH",
            "guidance": "Wind speeds exceeding 40 km/h present direct mechanical stability hazards.",
            "precautions": ["Avoid overpasses"]
        },
        "weather_data": {
            "temperature_2m": 25.0,
            "wind_speed_10m": 45.0,
            "precipitation": 0.0,
            "precipitation_probability": 10,
            "uv_index": 4.0,
            "weather_description": "Windy",
            "location_name": "Chennai"
        },
        "retry_count": 0,
    }

    # Validate output - should catch missing SOP-EXER-002 citation
    validated = validate_grounding_node(invalid_state)
    assert validated["grounding_valid"] is False
    assert len(validated["grounding_errors"]) > 0
    assert validated["retry_count"] == 1

    # Simulate reaching retry limit -> check routing routes to fallback
    invalid_state["retry_count"] = 2
    invalid_state["grounding_valid"] = False
    next_step = check_grounding_condition(invalid_state)
    assert next_step == "fallback"

    # Execute deterministic fallback node
    fallback_res = deterministic_fallback_node(invalid_state)
    assert fallback_res["route"] == "deterministic_fallback"
    assert "SOP-EXER-002" in fallback_res["final_response"]
    assert "45.0 km/h" in fallback_res["final_response"]
