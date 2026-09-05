import pytest
import json
import tempfile
import os
from src.sop_engine import SOPEngine, SEVERITY_RANKS


def test_sop_loading():
    engine = SOPEngine()
    sops = engine.load_sops()
    assert len(sops) >= 10
    categories = set(s["category"] for s in sops)
    assert len(categories) >= 3


def test_uv_exercise_matching():
    engine = SOPEngine()
    mock_weather = {
        "uv_index": 9.5,
        "temperature_2m": 30.0,
        "precipitation": 0.0,
        "wind_speed_10m": 10.0,
    }
    matched = engine.evaluate(
        weather=mock_weather,
        activity="running",
        demographics=["athletes"],
        timeframe="afternoon",
        raw_query="Can I go for a run in the afternoon?",
    )
    sop_ids = [s["id"] for s in matched]
    assert "SOP-EXER-001" in sop_ids


def test_wind_cycling_matching():
    engine = SOPEngine()
    mock_weather = {
        "uv_index": 3.0,
        "temperature_2m": 22.0,
        "precipitation": 0.0,
        "wind_speed_10m": 48.0,
    }
    matched = engine.evaluate(
        weather=mock_weather,
        activity="cycling",
        demographics=["all"],
        raw_query="Is it safe to bike to work with strong winds?",
    )
    assert len(matched) > 0
    assert matched[0]["id"] == "SOP-EXER-002"
    assert matched[0]["severity"] == "HIGH"


def test_fuzzy_picnic_comfort_index():
    engine = SOPEngine()
    # Ideal picnic conditions
    ideal_weather = {
        "temperature_2m": 22.0,
        "precipitation": 0.0,
        "precipitation_probability": 10,
        "wind_speed_10m": 12.0,
        "uv_index": 4.0,
    }
    matched = engine.evaluate(
        weather=ideal_weather,
        activity="picnic",
        raw_query="Is today good for a family picnic in the park?",
    )
    sop_ids = [s["id"] for s in matched]
    assert "SOP-LEISURE-001" in sop_ids


def test_severe_weather_override_priority():
    engine = SOPEngine()
    # Severe monsoonal conditions
    cyclone_weather = {
        "precipitation": 25.0,
        "precipitation_probability": 95,
        "wind_speed_10m": 52.0,
        "weather_code": 95,  # Thunderstorm
        "uv_index": 1.0,
        "temperature_2m": 26.0,
    }
    matched = engine.evaluate(
        weather=cyclone_weather,
        activity="cycling",
        raw_query="Can I cycle during this thunderstorm?",
    )
    assert len(matched) >= 1
    # SOP-OVERRIDE-001 should be primary due to CRITICAL severity
    assert matched[0]["id"] == "SOP-OVERRIDE-001"
    assert matched[0]["severity"] == "CRITICAL"


def test_multi_match_severity_sorting():
    engine = SOPEngine()
    # High UV (HIGH) and Moderate Commute Rain (MODERATE)
    mixed_weather = {
        "uv_index": 9.0,
        "precipitation_probability": 80,
        "precipitation": 4.0,
        "wind_speed_10m": 15.0,
        "temperature_2m": 32.0,
    }
    matched = engine.evaluate(
        weather=mixed_weather,
        activity="two-wheeler",
        timeframe="midday",
        raw_query="Riding scooter midday in the rain",
    )
    # Ensure items are ordered by severity descending
    ranks = [s["severity_rank"] for s in matched]
    assert ranks == sorted(ranks, reverse=True)


def test_zero_code_dynamic_sop_addition():
    """Verifies that an 11th/new SOP added to JSON is instantly recognized with ZERO code modifications."""
    engine = SOPEngine()
    current_sops = engine.load_sops()

    custom_sop = {
        "id": "SOP-SURF-001",
        "title": "Coastal Surfing & Rip Current Safety Advisory",
        "category": "outdoor_exercise",
        "target_activities": ["surfing", "bodyboarding", "surf"],
        "target_demographics": ["surfers", "all"],
        "severity": "HIGH",
        "conditions": {
            "type": "numeric",
            "rules": [
                {
                    "field": "wind_speed_10m",
                    "op": ">=",
                    "value": 30.0
                }
            ]
        },
        "guidance": "Gale-force offshore winds create hazardous rip currents and turbulent surf breaks.",
        "precautions": ["Surfers must wear a high-visibility leash and avoid solo sessions."]
    }

    # Write to a temporary SOP file to test dynamic loading
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as f:
        temp_path = f.name
        json.dump(current_sops + [custom_sop], f)

    try:
        temp_engine = SOPEngine(sops_path=temp_path)
        mock_surf_weather = {
            "wind_speed_10m": 38.0,
            "temperature_2m": 26.0,
            "precipitation": 0.0,
            "uv_index": 4.0,
        }
        matched = temp_engine.evaluate(
            weather=mock_surf_weather,
            activity="surfing",
            raw_query="Is it safe to go surfing in Goa today?",
        )
        assert any(s["id"] == "SOP-SURF-001" for s in matched)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
