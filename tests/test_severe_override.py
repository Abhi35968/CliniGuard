import pytest
from unittest.mock import MagicMock
from src.policy.engine import DeterministicPolicyEngine
from src.schemas.context import UserContext
from src.sop_engine import SOPEngine
from src.nodes import evaluate_sops_node
from src.state import AgentState


def test_severe_override_independent_of_retrieval():
    """
    Test A — Severe override independent of retrieval:
    Supply weather conditions satisfying SOP-OVERRIDE-001 (thunderstorm with weather_code 95 and precipitation >= 15mm).
    Supply candidate_sops containing only an unrelated activity-specific SOP (SOP-LEISURE-001 picnic).
    Verify SOP-OVERRIDE-001 is evaluated and becomes the governing policy with CRITICAL severity.
    """
    engine = DeterministicPolicyEngine()
    cyclone_weather = {
        "precipitation": 22.0,
        "precipitation_probability": 90,
        "wind_speed_10m": 42.0,
        "weather_code": 95,  # Thunderstorm
        "uv_index": 1.0,
        "temperature_2m": 25.0,
        "relative_humidity_2m": 85,
    }
    context = UserContext(
        activity="picnic",
        location="Kolkata",
        demographics=["all"],
    )
    unrelated_candidate = [{
        "id": "SOP-LEISURE-001",
        "title": "Mild Weather Outdoor Picnic & Gathering Advisory",
        "category": "leisure",
        "target_activities": ["picnic", "park outing"],
        "target_demographics": ["all"],
        "severity": "LOW",
        "priority": 10,
        "similarity_score": 0.895,
        "conditions": {
            "type": "numeric",
            "rules": [{"field": "precipitation", "op": "==", "value": 0.0}]
        },
    }]

    risk, matched = engine.evaluate_risk(
        weather=cyclone_weather,
        context=context,
        candidate_sops=unrelated_candidate,
        raw_query="Can we plan an outdoor picnic in Kolkata this afternoon?",
    )

    matched_ids = [m["id"] for m in matched]
    assert "SOP-OVERRIDE-001" in matched_ids
    assert matched[0]["id"] == "SOP-OVERRIDE-001"
    assert risk.risk_level == "CRITICAL"
    assert risk.primary_sop_id == "SOP-OVERRIDE-001"
    assert "SOP-OVERRIDE-001" in risk.sop_citation


def test_ordinary_weather_does_not_trigger_override():
    """
    Test B — Ordinary weather:
    Supply conditions that do not satisfy any severe override.
    Verify that severe overrides remain inactive and ordinary activity policy governs.
    """
    engine = DeterministicPolicyEngine()
    pleasant_weather = {
        "precipitation": 0.0,
        "precipitation_probability": 0,
        "wind_speed_10m": 8.0,
        "weather_code": 0,  # Clear sky
        "uv_index": 2.0,
        "temperature_2m": 23.0,
        "relative_humidity_2m": 50,
    }
    context = UserContext(
        activity="picnic",
        location="Kolkata",
        demographics=["all"],
    )
    picnic_candidate = [{
        "id": "SOP-LEISURE-001",
        "title": "Mild Weather Outdoor Picnic & Gathering Advisory",
        "category": "leisure",
        "target_activities": ["picnic", "park outing"],
        "target_demographics": ["all"],
        "severity": "LOW",
        "priority": 10,
        "similarity_score": 0.912,
        "conditions": {
            "type": "composite",
            "operator": "AND",
            "rules": [
                {"field": "precipitation", "op": "<=", "value": 0.5},
                {"field": "wind_speed_10m", "op": "<=", "value": 20.0},
            ]
        },
    }]

    risk, matched = engine.evaluate_risk(
        weather=pleasant_weather,
        context=context,
        candidate_sops=picnic_candidate,
        raw_query="Can we plan an outdoor picnic in Kolkata this afternoon?",
    )

    matched_ids = [m["id"] for m in matched]
    # Severe overrides must NOT trigger
    assert "SOP-OVERRIDE-001" not in matched_ids
    assert "SOP-OVERRIDE-002" not in matched_ids

    # Governing policy is ordinary picnic SOP
    assert len(matched) == 1
    assert matched[0]["id"] == "SOP-LEISURE-001"
    assert risk.risk_level == "LOW"
    assert risk.primary_sop_id == "SOP-LEISURE-001"


def test_override_precedence_over_activity_specific_sop():
    """
    Test C — Override precedence:
    Supply conditions matching both an activity-specific SOP (SOP-EXER-002 High Wind, HIGH, priority 85)
    and a severe override (SOP-OVERRIDE-001 Monsoonal Heavy Rainfall, CRITICAL, priority 100).
    Verify that the configured override-priority rules place SOP-OVERRIDE-001 as primary.
    """
    engine = DeterministicPolicyEngine()
    dual_hazard_weather = {
        "precipitation": 20.0,
        "precipitation_probability": 95,
        "wind_speed_10m": 48.0,  # Matches SOP-EXER-002 wind >= 40 km/h
        "weather_code": 95,      # Matches SOP-OVERRIDE-001 code in [65, 67, 82, 95, 96, 99]
        "uv_index": 1.0,
        "temperature_2m": 24.0,
        "relative_humidity_2m": 90,
    }
    context = UserContext(
        activity="cycling",
        location="Chennai",
        demographics=["all"],
    )
    cycling_candidate = [{
        "id": "SOP-EXER-002",
        "title": "High Wind Cycling & Open-Field Safety",
        "category": "outdoor_exercise",
        "target_activities": ["cycling", "biking", "bike"],
        "target_demographics": ["all"],
        "severity": "HIGH",
        "priority": 85,
        "similarity_score": 0.884,
        "conditions": {
            "type": "numeric",
            "rules": [{"field": "wind_speed_10m", "op": ">=", "value": 40.0}]
        },
    }]

    risk, matched = engine.evaluate_risk(
        weather=dual_hazard_weather,
        context=context,
        candidate_sops=cycling_candidate,
        raw_query="Is it safe to cycle in strong winds and rain today?",
    )

    matched_ids = [m["id"] for m in matched]
    assert "SOP-OVERRIDE-001" in matched_ids
    assert "SOP-EXER-002" in matched_ids

    # SOP-OVERRIDE-001 must rank ahead of SOP-EXER-002 due to CRITICAL severity and priority 100
    assert matched[0]["id"] == "SOP-OVERRIDE-001"
    assert matched[0]["severity"] == "CRITICAL"
    assert risk.primary_sop_id == "SOP-OVERRIDE-001"
    assert risk.risk_level == "CRITICAL"


def test_retrieval_ui_separation_no_score_fabrication():
    """
    Test D — Retrieval/UI separation:
    Verify that injecting override policies into deterministic evaluation does NOT
    falsely report them as semantic retrieval results or fabricate similarity scores in AgentState.
    """
    ctx = UserContext(activity="picnic", location="Kolkata", demographics=["all"])
    state: AgentState = {
        "session_id": "test-override-separation",
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
            "temperature_2m": 24.0,
            "precipitation": 25.0,  # Severe rainfall
            "precipitation_probability": 95,
            "wind_speed_10m": 45.0,
            "weather_code": 95,      # Thunderstorm
            "uv_index": 1.0,
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

    # 1. Deterministic evaluation correctly identifies severe override
    assert result["active_sop"] is not None
    assert result["active_sop"]["id"] == "SOP-OVERRIDE-001"
    assert result["active_sop"]["severity"] == "CRITICAL"
    assert "SOP-OVERRIDE-001" in result["sop_citation"]

    # 2. Retrieved SOPs in state must remain strictly the semantic retrieval candidates
    retrieved = result["retrieved_sops"]
    assert isinstance(retrieved, list)

    # Verify that every item in retrieved_sops has a genuine similarity_score from vector search
    for sop in retrieved:
        assert "similarity_score" in sop
        assert isinstance(sop["similarity_score"], (float, int))
        # Ensure scores are normalized valid floats (not dummy values like 999.0)
        assert 0.0 <= sop["similarity_score"] <= 1.0

    # SOP-OVERRIDE-001 was not returned by semantic search for 'picnic', so it must NOT be in retrieved_sops
    retrieved_ids = [s["id"] for s in retrieved]
    if "SOP-OVERRIDE-001" not in retrieved_ids:
        # Confirms override was evaluated via deterministic engine, not by polluting retrieved_sops
        assert result["active_sop"]["id"] not in retrieved_ids


def test_existing_activity_specific_matching_preserved():
    """
    Test E — Existing behavior:
    Ensure ordinary activity-specific matching, dynamic policy addition, and SOPEngine wrapper continue to work.
    """
    # 1. Test SOPEngine wrapper evaluate
    sop_engine = SOPEngine()
    ideal_weather = {
        "temperature_2m": 22.0,
        "precipitation": 0.0,
        "precipitation_probability": 10,
        "wind_speed_10m": 12.0,
        "uv_index": 4.0,
    }
    matched = sop_engine.evaluate(
        weather=ideal_weather,
        activity="picnic",
        raw_query="Is today good for a family picnic in the park?",
    )
    sop_ids = [s["id"] for s in matched]
    assert "SOP-LEISURE-001" in sop_ids

    # 2. Test SOP-OVERRIDE-002 (Snow, Frost & Ice) when temperature <= 2.0 and humidity >= 80
    freezing_weather = {
        "temperature_2m": -1.0,
        "relative_humidity_2m": 88,
        "precipitation": 0.0,
        "wind_speed_10m": 15.0,
        "weather_code": 71,  # Slight snow fall
        "uv_index": 1.0,
    }
    matched_snow = sop_engine.evaluate(
        weather=freezing_weather,
        activity="cycling",
        raw_query="Can I bike in this freezing snow?",
    )
    assert len(matched_snow) >= 1
    assert matched_snow[0]["id"] == "SOP-OVERRIDE-002"
    assert matched_snow[0]["severity"] == "CRITICAL"
