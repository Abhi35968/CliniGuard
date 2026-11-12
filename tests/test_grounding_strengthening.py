"""
Regression test suite for CliniGuard Strengthened Output Grounding Guardrail.

Covers:
- Test A: CRITICAL risk with explicitly permissive advisory ("It is safe to proceed") -> rejected.
- Test B: HIGH risk with recommendation violating configured prohibition -> rejected.
- Test C: Correct warning stating activity is NOT safe (negated safety) -> passes (no false positives).
- Test D: Missing mandatory risk warning or restriction under high/critical risk -> rejected.
- Test E: Correct grounded advisories (both high-risk and low-risk) -> pass.
- Test F: Validation failure followed by deterministic fallback -> draft invalidated and fallback returned.
- Test G: Existing numeric consistency and SOP citation checks remain intact.
- Test H: Severe weather override grounding regression -> ensures override invariants are upheld.
"""

import pytest
from src.guardrails.grounding import GroundingGuardrail
from src.guardrails.output import OutputGuardrailPipeline
from src.schemas.risk import RiskAssessment
from src.policy.engine import DeterministicPolicyEngine
from src.schemas.context import UserContext
from src.nodes import (
    validate_grounding_node,
    deterministic_fallback_node,
    check_grounding_condition,
)


@pytest.fixture
def guardrail():
    return GroundingGuardrail()


@pytest.fixture
def critical_override_sop():
    return {
        "id": "SOP-OVERRIDE-001",
        "title": "Cyclone, Gale-Force Winds & Extreme Storm Override",
        "category": "override",
        "severity": "CRITICAL",
        "target_activities": ["all"],
        "target_demographics": ["all"],
        "guidance": "Severe storm conditions detected. Immediate indoor shelter mandatory.",
        "precautions": ["Shelter indoors", "Secure loose objects"],
        "prohibited_actions": [
            "All outdoor activities",
            "Movement near trees, utility lines, and temporary structures",
            "Operation of two-wheelers and high-profile vehicles",
        ],
    }


@pytest.fixture
def high_wind_sop():
    return {
        "id": "SOP-EXER-002",
        "title": "Squally Wind & Crosswind Hazard",
        "category": "exercise",
        "severity": "HIGH",
        "target_activities": ["cycling", "two-wheeler"],
        "target_demographics": ["all"],
        "guidance": "Wind speeds exceeding 40 km/h present direct mechanical stability hazards.",
        "precautions": ["Postpone outdoor cycling", "Use public transit"],
        "prohibited_actions": [
            "Cycling on elevated bridges, overpasses, or coastal causeways",
            "Non-essential two-wheeler transit during peak gusts",
        ],
    }


@pytest.fixture
def low_risk_sop():
    return {
        "id": "SOP-LEISURE-001",
        "title": "Mild Weather Outdoor Picnic & Gathering Advisory",
        "category": "leisure",
        "severity": "LOW",
        "target_activities": ["picnic", "park outing"],
        "target_demographics": ["all"],
        "guidance": "Conditions are favorable for outdoor leisure activities.",
        "precautions": ["Standard hydration"],
        "prohibited_actions": [],
    }


# =========================================================================
# Test A: CRITICAL risk with an explicitly permissive advisory
# =========================================================================
def test_a_critical_risk_rejects_explicitly_permissive_advisory(guardrail, critical_override_sop):
    """
    CRITICAL risk with an explicitly permissive advisory such as 'It is safe to proceed.'
    The validator must reject the contradiction.
    """
    risk = RiskAssessment(
        risk_level="CRITICAL",
        primary_sop_id="SOP-OVERRIDE-001",
        prohibited_actions=critical_override_sop["prohibited_actions"],
        sop_citation="SOP-OVERRIDE-001",
    )
    weather = {
        "temperature_2m": 24.0,
        "wind_speed_10m": 55.0,
        "precipitation": 25.0,
    }

    permissive_advisories = [
        (
            "SOP-OVERRIDE-001: Winds are 55.0 km/h with 25.0 mm precipitation. "
            "It is safe to proceed with your outdoor walk."
        ),
        (
            "Under SOP-OVERRIDE-001 (55.0 km/h winds, 25.0 mm precipitation), "
            "it is completely safe to go outside today."
        ),
        (
            "According to SOP-OVERRIDE-001, temperature is 24.0°C and wind is 55.0 km/h. "
            "You are all clear and there is no risk to your journey."
        ),
        (
            "SOP-OVERRIDE-001 Telemetry: 55.0 km/h winds, 25.0 mm rain. "
            "STATUS: APPROVED. Severity Level: LOW. Have a great picnic!"
        ),
    ]

    for adv in permissive_advisories:
        result = guardrail.validate_grounding(
            response_text=adv,
            weather_data=weather,
            active_sop=critical_override_sop,
            risk_assessment=risk,
        )
        assert result.passed is False, f"Expected advisory to be rejected: {adv}"
        assert len(result.violations) > 0
        assert any(
            "permissive" in v.lower() or "downgrade" in v.lower() or "contradiction" in v.lower()
            for v in result.violations
        )


# =========================================================================
# Test B: HIGH risk with a recommendation that violates an actual prohibition
# =========================================================================
def test_b_high_risk_rejects_recommendation_violating_prohibition(guardrail, high_wind_sop):
    """
    HIGH risk with a recommendation that violates an actual configured prohibition.
    The validator must reject it.
    """
    risk = RiskAssessment(
        risk_level="HIGH",
        primary_sop_id="SOP-EXER-002",
        prohibited_actions=high_wind_sop["prohibited_actions"],
        sop_citation="SOP-EXER-002",
    )
    weather = {
        "wind_speed_10m": 45.0,
        "temperature_2m": 22.0,
        "precipitation": 0.0,
    }

    violating_advisories = [
        (
            "WARNING: HIGH RISK under SOP-EXER-002 due to 45.0 km/h winds. "
            "We recommend cycling on overpasses to stay above traffic."
        ),
        (
            "CAUTION under SOP-EXER-002: Wind speed is 45.0 km/h. "
            "You can take your two-wheeler transit if you are careful."
        ),
        (
            "HIGH RISK SOP-EXER-002 notice: 45.0 km/h winds. "
            "Feel free to ride across elevated bridges during low traffic periods."
        ),
    ]

    for adv in violating_advisories:
        result = guardrail.validate_grounding(
            response_text=adv,
            weather_data=weather,
            active_sop=high_wind_sop,
            risk_assessment=risk,
        )
        assert result.passed is False, f"Expected prohibition violation rejection: {adv}"
        assert any("prohibited" in v.lower() for v in result.violations)


# =========================================================================
# Test C: Correct warning stating activity is NOT safe
# =========================================================================
def test_c_correct_warning_with_negated_safe_is_not_rejected(guardrail, critical_override_sop):
    """
    A correct warning that says an activity is NOT safe.
    It must not be rejected simply because the text contains the word 'safe'.
    """
    risk = RiskAssessment(
        risk_level="CRITICAL",
        primary_sop_id="SOP-OVERRIDE-001",
        prohibited_actions=critical_override_sop["prohibited_actions"],
        sop_citation="SOP-OVERRIDE-001",
    )
    weather = {
        "wind_speed_10m": 55.0,
        "precipitation": 25.0,
        "temperature_2m": 24.0,
    }

    valid_warning_advisories = [
        (
            "CRITICAL HAZARD WARNING: Under SOP-OVERRIDE-001 with 55.0 km/h winds and 25.0 mm rain, "
            "it is NOT safe to proceed with any outdoor activity. Do not go outside. "
            "For your personal safety, shelter indoors immediately."
        ),
        (
            "DANGER: Under SOP-OVERRIDE-001, wind speed is 55.0 km/h and precipitation is 25.0 mm. "
            "Outdoor exercise is never safe during extreme storms. All outdoor activities are strictly prohibited."
        ),
        (
            "CRITICAL: SOP-OVERRIDE-001 applies with 55.0 km/h winds and 25.0 mm precipitation. "
            "It cannot be considered safe to cycle or walk. Postpone all outdoor plans and remain indoors."
        ),
    ]

    for adv in valid_warning_advisories:
        result = guardrail.validate_grounding(
            response_text=adv,
            weather_data=weather,
            active_sop=critical_override_sop,
            risk_assessment=risk,
        )
        assert result.passed is True, f"Legitimate warning was incorrectly rejected: {result.violations}"
        assert len(result.violations) == 0


# =========================================================================
# Test D: Missing mandatory risk warning or restriction
# =========================================================================
def test_d_missing_mandatory_risk_warning_rejected(guardrail, critical_override_sop, high_wind_sop):
    """
    Missing mandatory risk warning or restriction.
    Reject it when the policy requires it.
    """
    # 1. Under CRITICAL risk: Omission of critical hazard warning tokens
    critical_risk = RiskAssessment(
        risk_level="CRITICAL",
        primary_sop_id="SOP-OVERRIDE-001",
        prohibited_actions=critical_override_sop["prohibited_actions"],
        sop_citation="SOP-OVERRIDE-001",
    )
    weather_crit = {
        "wind_speed_10m": 55.0,
        "precipitation": 25.0,
    }
    # Advisory has numbers and citation, but no hazard/prohibition/warning language
    casual_advisory = (
        "Under SOP-OVERRIDE-001, recorded wind speed is 55.0 km/h and precipitation is 25.0 mm. "
        "The weather is quite wet and blustery outside."
    )
    res_crit = guardrail.validate_grounding(
        response_text=casual_advisory,
        weather_data=weather_crit,
        active_sop=critical_override_sop,
        risk_assessment=critical_risk,
    )
    assert res_crit.passed is False
    assert any("mandatory warning" in v.lower() for v in res_crit.violations)

    # 2. Under HIGH risk: Omission of high-risk warning/caution tokens
    high_risk = RiskAssessment(
        risk_level="HIGH",
        primary_sop_id="SOP-EXER-002",
        prohibited_actions=high_wind_sop["prohibited_actions"],
        sop_citation="SOP-EXER-002",
    )
    weather_high = {
        "wind_speed_10m": 45.0,
    }
    neutral_high_advisory = (
        "Regarding SOP-EXER-002, wind speed is 45.0 km/h today. "
        "Dress warmly if you go outdoors."
    )
    res_high = guardrail.validate_grounding(
        response_text=neutral_high_advisory,
        weather_data=weather_high,
        active_sop=high_wind_sop,
        risk_assessment=high_risk,
    )
    assert res_high.passed is False
    assert any("mandatory warning" in v.lower() for v in res_high.violations)


# =========================================================================
# Test E: Correct grounded advisories pass
# =========================================================================
def test_e_correct_grounded_advisory_passes(guardrail, high_wind_sop, low_risk_sop):
    """
    A correct, grounded advisory must pass.
    Tests both HIGH severity and LOW severity with verified telemetry.
    """
    # High risk with accurate warning, citation, telemetry, and restrictions
    risk_high = RiskAssessment(
        risk_level="HIGH",
        primary_sop_id="SOP-EXER-002",
        prohibited_actions=high_wind_sop["prohibited_actions"],
        sop_citation="SOP-EXER-002",
    )
    weather_high = {
        "wind_speed_10m": 45.0,
        "temperature_2m": 22.0,
        "precipitation": 0.0,
    }
    advisory_high = (
        "HIGH RISK WARNING under SOP-EXER-002: Wind speed is currently 45.0 km/h with 22.0°C temperature. "
        "Non-essential two-wheeler transit and cycling on overpasses are strictly prohibited. "
        "Please avoid outdoor cycling until conditions improve."
    )
    res_high = guardrail.validate_grounding(
        response_text=advisory_high,
        weather_data=weather_high,
        active_sop=high_wind_sop,
        risk_assessment=risk_high,
    )
    assert res_high.passed is True
    assert len(res_high.violations) == 0

    # Low risk advisory
    risk_low = RiskAssessment(
        risk_level="LOW",
        primary_sop_id="SOP-LEISURE-001",
        prohibited_actions=[],
        sop_citation="SOP-LEISURE-001",
    )
    weather_low = {
        "temperature_2m": 21.5,
        "precipitation": 0.0,
        "wind_speed_10m": 8.0,
    }
    advisory_low = (
        "Under SOP-LEISURE-001, temperature is 21.5°C with 0.0 mm precipitation and 8.0 km/h wind. "
        "Conditions are favorable for your picnic. Enjoy your time outdoors."
    )
    res_low = guardrail.validate_grounding(
        response_text=advisory_low,
        weather_data=weather_low,
        active_sop=low_risk_sop,
        risk_assessment=risk_low,
    )
    assert res_low.passed is True
    assert len(res_low.violations) == 0


# =========================================================================
# Test F: Validation failure followed by deterministic fallback
# =========================================================================
def test_f_validation_failure_leads_to_fallback_without_draft_leak(critical_override_sop):
    """
    Confirm that when validation fails, the rejected draft is invalidated
    (state['final_response'] is cleared), and after retries exhaust,
    the deterministic fallback node produces the final response (not the rejected draft).
    """
    hallucinated_draft = (
        "Don't worry about the storm! It is safe to proceed outside! "
        "SOP-OVERRIDE-001 winds are 55.0 km/h and rain is 25.0 mm, but you will be fine."
    )

    state = {
        "final_response": hallucinated_draft,
        "active_sop": critical_override_sop,
        "risk_assessment": RiskAssessment(
            risk_level="CRITICAL",
            primary_sop_id="SOP-OVERRIDE-001",
            prohibited_actions=critical_override_sop["prohibited_actions"],
            sop_citation="SOP-OVERRIDE-001",
        ),
        "weather_data": {
            "temperature_2m": 24.0,
            "wind_speed_10m": 55.0,
            "precipitation": 25.0,
            "precipitation_probability": 90,
            "uv_index": 1.0,
            "weather_description": "Thunderstorm",
            "location_name": "Kolkata",
        },
        "context": UserContext(
            activity="walk",
            location="Kolkata",
            demographics=["all"],
        ),
        "retry_count": 0,
    }

    # Step 1: Run validate_grounding_node
    validated_state = validate_grounding_node(state)
    assert validated_state["grounding_valid"] is False
    assert len(validated_state["grounding_errors"]) > 0
    # Crucial safety check: the rejected draft MUST NOT remain in final_response!
    assert validated_state["final_response"] is None

    # Merge updates into state (simulating LangGraph state update)
    state.update(validated_state)

    # Step 2: Retry exhaustion check
    state["retry_count"] = 2
    next_step = check_grounding_condition(state)
    assert next_step == "fallback"

    # Step 3: Run deterministic fallback node
    fallback_state = deterministic_fallback_node(state)
    assert fallback_state["route"] == "deterministic_fallback"
    assert fallback_state["final_response"] is not None
    # Confirm rejected text did not leak
    assert "It is safe to proceed" not in fallback_state["final_response"]
    assert "you will be fine" not in fallback_state["final_response"]
    # Confirm fallback contains authoritative deterministic safety guidance
    assert "CRITICAL" in fallback_state["final_response"]
    assert "SOP-OVERRIDE-001" in fallback_state["final_response"]
    assert "55.0 km/h" in fallback_state["final_response"]

    # Step 4: The deterministic fallback itself must pass validation
    guard = GroundingGuardrail()
    val_fallback = guard.validate_grounding(
        response_text=fallback_state["final_response"],
        weather_data=state["weather_data"],
        active_sop=critical_override_sop,
        risk_assessment=state["risk_assessment"],
    )
    assert val_fallback.passed is True, f"Fallback failed validation: {val_fallback.violations}"


# =========================================================================
# Test G: Existing numeric consistency and SOP citation behavior remain intact
# =========================================================================
def test_g_numeric_consistency_and_sop_citation_checks_intact(guardrail, high_wind_sop):
    """
    Ensure existing numeric consistency checks and SOP citation checks
    continue to function as expected.
    """
    risk = RiskAssessment(
        risk_level="HIGH",
        primary_sop_id="SOP-EXER-002",
        prohibited_actions=high_wind_sop["prohibited_actions"],
        sop_citation="SOP-EXER-002",
    )
    weather = {
        "wind_speed_10m": 45.0,
        "temperature_2m": 22.0,
        "precipitation": 0.0,
    }

    # Case 1: Hallucinated number (advisory claims 12.0 km/h wind when telemetry is 45.0 km/h)
    hallucinated_num_adv = (
        "HIGH RISK CAUTION under SOP-EXER-002: Wind speed is only 12.0 km/h. "
        "Avoid overpasses."
    )
    pipeline = OutputGuardrailPipeline()
    passed, results = pipeline.validate_output(
        response_text=hallucinated_num_adv,
        risk_assessment=risk,
        matched_sops=[high_wind_sop],
        telemetry=weather,
    )
    assert passed is False
    all_violations = [v for r in results for v in r.violations]
    assert any("telemetry" in v.lower() or "numeric" in v.lower() or "45.0" in v for v in all_violations)

    # Case 2: Missing SOP citation
    missing_sop_adv = (
        "HIGH RISK CAUTION: Wind speed is 45.0 km/h. "
        "Avoid cycling on overpasses."
    )
    res_sop = guardrail.validate_grounding(
        response_text=missing_sop_adv,
        weather_data=weather,
        active_sop=high_wind_sop,
        risk_assessment=risk,
    )
    assert res_sop.passed is False
    assert any("sop" in v.lower() and "citation" in v.lower() for v in res_sop.violations)

    # Case 3: Invents a non-existent SOP
    invented_sop_adv = (
        "HIGH RISK CAUTION under SOP-EXER-002 and SOP-MAGIC-999: Wind speed is 45.0 km/h. "
        "Avoid cycling on overpasses."
    )
    res_inv = guardrail.validate_grounding(
        response_text=invented_sop_adv,
        weather_data=weather,
        active_sop=high_wind_sop,
        risk_assessment=risk,
    )
    assert res_inv.passed is False
    assert any("invented" in v.lower() or "hallucinated" in v.lower() or "magic" in v.lower() for v in res_inv.violations)


# =========================================================================
# Test H: Severe weather override integration with grounding validation
# =========================================================================
def test_h_severe_weather_override_grounding_integration(guardrail):
    """
    Evaluates policy engine severe override under extreme cyclone weather,
    verifying that when the override triggers, the grounding guardrail
    strictly enforces the override's CRITICAL invariants.
    """
    engine = DeterministicPolicyEngine()
    cyclone_weather = {
        "precipitation": 28.0,
        "precipitation_probability": 95,
        "wind_speed_10m": 62.0,
        "weather_code": 95,  # Thunderstorm
        "uv_index": 1.0,
        "temperature_2m": 25.0,
        "relative_humidity_2m": 90,
    }
    context = UserContext(
        activity="picnic",
        location="Kolkata",
        demographics=["all"],
    )

    risk, matched = engine.evaluate_risk(
        weather=cyclone_weather,
        context=context,
        candidate_sops=[],
        raw_query="Can we have an outdoor picnic in Kolkata?",
    )

    assert risk.risk_level == "CRITICAL"
    assert risk.primary_sop_id == "SOP-OVERRIDE-001"
    active_sop = matched[0]

    # Contradicting advisory claiming weather is suitable for picnic
    bad_adv = (
        "According to SOP-OVERRIDE-001, weather shows 62.0 km/h winds and 28.0 mm rain. "
        "It is safe to proceed with your outdoor picnic."
    )
    res_bad = guardrail.validate_grounding(
        response_text=bad_adv,
        weather_data=cyclone_weather,
        active_sop=active_sop,
        risk_assessment=risk,
    )
    assert res_bad.passed is False
    assert any("permissive" in v.lower() or "contradiction" in v.lower() for v in res_bad.violations)

    # Grounded warning adhering to override prohibitions
    good_adv = (
        "CRITICAL STORM OVERRIDE WARNING: Governing policy SOP-OVERRIDE-001 is in effect. "
        "Recorded telemetry: wind speed 62.0 km/h, precipitation 28.0 mm. "
        "All outdoor activities are strictly prohibited due to gale-force winds and storm hazards. "
        "Postpone your picnic immediately and shelter indoors."
    )
    res_good = guardrail.validate_grounding(
        response_text=good_adv,
        weather_data=cyclone_weather,
        active_sop=active_sop,
        risk_assessment=risk,
    )
    assert res_good.passed is True
    assert len(res_good.violations) == 0
