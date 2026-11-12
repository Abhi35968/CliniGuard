"""
Regression test suite for CliniGuard Weather Timeframe Aggregation (Finding 3).

Covers:
- Test A: Today's afternoon: 0 mm rainfall today, 20 mm rainfall tomorrow. "this afternoon" only selects today.
- Test B: Tomorrow's afternoon: Same fixture, "tomorrow afternoon" selects tomorrow's values (20 mm).
- Test C: Morning date selection: Today's morning vs tomorrow's morning isolated without cross-date mixing.
- Test D: Date boundary and timezone: Timezone-aware UTC timestamps crossing date boundaries mapped to local date/hour.
- Test E: Incomplete or missing data: Missing hours/None values handled safely without fabricated zeroes.
- Test F: Backward compatibility: Calls without timeframe return standard current weather metrics.
- Test G: Downstream safety: Correct telemetry flows to DeterministicPolicyEngine, verifying risk assessment accuracy.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from src.tools.weather import WeatherService, WeatherTool, get_weather_description
from src.policy.engine import DeterministicPolicyEngine
from src.schemas.context import UserContext


def _generate_mock_open_meteo_response(
    start_date: str = "2026-10-20",
    today_afternoon_rain: float = 0.0,
    tomorrow_afternoon_rain: float = 20.0,
    today_afternoon_temp: float = 25.0,
    tomorrow_afternoon_temp: float = 18.0,
    timezone: str = "Asia/Kolkata",
) -> dict:
    """Generates 48-hour mock Open-Meteo response spanning today and tomorrow."""
    base_dt = datetime.fromisoformat(f"{start_date}T00:00:00")
    timestamps = [(base_dt + timedelta(hours=i)).strftime("%Y-%m-%dT%H:00") for i in range(48)]

    # 48 hours of baseline metrics
    temperatures = [20.0] * 48
    precip_prob = [0] * 48
    precipitation = [0.0] * 48
    wind_speeds = [10.0] * 48
    uv_indices = [2.0] * 48
    weather_codes = [0] * 48

    # Populate Day 1 (today) afternoon: hours 12..16 (indices 12..16)
    for h in range(12, 17):
        temperatures[h] = today_afternoon_temp
        precipitation[h] = today_afternoon_rain
        precip_prob[h] = 0 if today_afternoon_rain == 0 else 80
        wind_speeds[h] = 12.0
        uv_indices[h] = 5.0
        weather_codes[h] = 0 if today_afternoon_rain == 0 else 61

    # Populate Day 2 (tomorrow) afternoon: hours 12..16 (indices 36..40)
    for h in range(36, 41):
        temperatures[h] = tomorrow_afternoon_temp
        precipitation[h] = tomorrow_afternoon_rain
        precip_prob[h] = 95 if tomorrow_afternoon_rain > 0 else 0
        wind_speeds[h] = 35.0
        uv_indices[h] = 3.0
        weather_codes[h] = 95 if tomorrow_afternoon_rain >= 15.0 else 63

    return {
        "latitude": 22.5726,
        "longitude": 88.3639,
        "timezone": timezone,
        "current": {
            "time": f"{start_date}T10:00",
            "temperature_2m": 22.0,
            "apparent_temperature": 23.0,
            "relative_humidity_2m": 60,
            "precipitation": 0.0,
            "precipitation_probability": 0,
            "wind_speed_10m": 10.0,
            "wind_gusts_10m": 14.0,
            "uv_index": 3.0,
            "cloud_cover": 10,
            "weather_code": 0,
        },
        "hourly": {
            "time": timestamps,
            "temperature_2m": temperatures,
            "apparent_temperature": temperatures,
            "relative_humidity_2m": [60] * 48,
            "precipitation_probability": precip_prob,
            "precipitation": precipitation,
            "wind_speed_10m": wind_speeds,
            "uv_index": uv_indices,
            "weather_code": weather_codes,
        },
    }


# =========================================================================
# Test A: Today's afternoon
# =========================================================================
def test_a_today_afternoon_selects_only_current_date():
    """
    Test A — Today's afternoon:
    - Today's afternoon has 0 mm rainfall.
    - Tomorrow's afternoon has 20 mm rainfall.
    - Request 'this afternoon'.
    - Verify only today's afternoon contributes to aggregated precipitation (0 mm, NOT 20 mm).
    """
    service = WeatherService()
    mock_data = _generate_mock_open_meteo_response(
        start_date="2026-10-20",
        today_afternoon_rain=0.0,
        tomorrow_afternoon_rain=20.0,
        today_afternoon_temp=26.5,
        tomorrow_afternoon_temp=17.5,
    )

    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_data
        mock_get.return_value = mock_resp

        parsed, err = service.fetch_live_weather(
            latitude=22.57, longitude=88.36, location_name="Kolkata", timeframe="this afternoon"
        )

        assert err is None
        assert parsed is not None
        # Must be today's 0.0 mm rainfall, NOT contaminated by tomorrow's 20.0 mm
        assert parsed["timeframe_precipitation"] == 0.0
        assert parsed["timeframe_precipitation_probability"] == 0
        assert parsed["timeframe_temperature_2m"] == 26.5
        assert parsed["timeframe_requested"] == "this afternoon"


# =========================================================================
# Test B: Tomorrow's afternoon
# =========================================================================
def test_b_tomorrow_afternoon_selects_only_following_date():
    """
    Test B — Tomorrow's afternoon:
    - Use the exact same fixture.
    - Request 'tomorrow afternoon'.
    - Verify tomorrow's values are selected (20 mm rainfall, 17.5°C).
    """
    service = WeatherService()
    mock_data = _generate_mock_open_meteo_response(
        start_date="2026-10-20",
        today_afternoon_rain=0.0,
        tomorrow_afternoon_rain=20.0,
        today_afternoon_temp=26.5,
        tomorrow_afternoon_temp=17.5,
    )

    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_data
        mock_get.return_value = mock_resp

        parsed, err = service.fetch_live_weather(
            latitude=22.57, longitude=88.36, location_name="Kolkata", timeframe="tomorrow afternoon"
        )

        assert err is None
        assert parsed is not None
        # Must select tomorrow's afternoon metrics
        assert parsed["timeframe_precipitation"] == 20.0
        assert parsed["timeframe_precipitation_probability"] == 95
        assert parsed["timeframe_temperature_2m"] == 17.5
        assert parsed["timeframe_wind_speed_10m"] == 35.0
        assert parsed["timeframe_requested"] == "tomorrow afternoon"


# =========================================================================
# Test C: Morning date selection
# =========================================================================
def test_c_morning_date_selection_isolated():
    """
    Test C — Morning date selection:
    - Verify today's morning and tomorrow's morning are not mixed.
    - Today's morning: wind 8 km/h, 0 mm rain.
    - Tomorrow's morning: wind 45 km/h, 12 mm rain.
    """
    service = WeatherService()
    mock_data = _generate_mock_open_meteo_response(start_date="2026-10-20")

    # Day 1 morning: hours 6..11
    for h in range(6, 12):
        mock_data["hourly"]["wind_speed_10m"][h] = 8.0
        mock_data["hourly"]["precipitation"][h] = 0.0

    # Day 2 morning: hours 30..35
    for h in range(30, 36):
        mock_data["hourly"]["wind_speed_10m"][h] = 45.0
        mock_data["hourly"]["precipitation"][h] = 12.0

    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_data
        mock_get.return_value = mock_resp

        # Request today's morning
        today_res, _ = service.fetch_live_weather(
            latitude=22.57, longitude=88.36, timeframe="this morning"
        )
        assert today_res["timeframe_wind_speed_10m"] == 8.0
        assert today_res["timeframe_precipitation"] == 0.0

        # Request tomorrow's morning
        tom_res, _ = service.fetch_live_weather(
            latitude=22.57, longitude=88.36, timeframe="tomorrow morning"
        )
        assert tom_res["timeframe_wind_speed_10m"] == 45.0
        assert tom_res["timeframe_precipitation"] == 12.0


# =========================================================================
# Test D: Date boundary and timezone
# =========================================================================
def test_d_date_boundary_and_timezone_respects_forecast_location():
    """
    Test D — Date boundary and timezone:
    - Verify local forecast dates are respected when timestamps cross UTC/local-date boundaries.
    - Kolkata is UTC+05:30.
    - Timestamps provided in UTC format with 'Z' suffix:
      e.g. 2026-10-20T06:30Z to 10:30Z correspond to 12:00 to 16:00 in Kolkata (afternoon of 2026-10-20).
      e.g. 2026-10-20T18:30Z corresponds to 2026-10-21T00:00+05:30 (midnight of tomorrow).
    """
    service = WeatherService()

    # Generate 48 hours in UTC timestamps
    utc_base = datetime(2026, 10, 20, 0, 0, tzinfo=ZoneInfo("UTC"))
    utc_timestamps = [(utc_base + timedelta(hours=i)).strftime("%Y-%m-%dT%H:00Z") for i in range(48)]

    # Find the UTC hours corresponding to local Kolkata afternoon on 2026-10-20 (12:00 to 16:00 local)
    # Kolkata is +05:30, so local 12:00 is UTC 06:30.
    indices = service._get_timeframe_indices(
        timestamps=utc_timestamps,
        timeframe="this afternoon",
        reference_time="2026-10-20T10:00:00+05:30",
        timezone_str="Asia/Kolkata",
    )

    # Verify that all selected indices map to local date 2026-10-20 and local hours 12..16
    assert len(indices) > 0
    for idx in indices:
        ts = utc_timestamps[idx]
        dt_local = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(ZoneInfo("Asia/Kolkata"))
        assert dt_local.date() == date(2026, 10, 20)
        assert 12 <= dt_local.hour <= 16

    # Verify tomorrow afternoon indices map to local date 2026-10-21
    tom_indices = service._get_timeframe_indices(
        timestamps=utc_timestamps,
        timeframe="tomorrow afternoon",
        reference_time="2026-10-20T10:00:00+05:30",
        timezone_str="Asia/Kolkata",
    )
    assert len(tom_indices) > 0
    for idx in tom_indices:
        ts = utc_timestamps[idx]
        dt_local = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(ZoneInfo("Asia/Kolkata"))
        assert dt_local.date() == date(2026, 10, 21)
        assert 12 <= dt_local.hour <= 16


# =========================================================================
# Test E: Incomplete or missing data
# =========================================================================
def test_e_incomplete_or_missing_forecast_data():
    """
    Test E — Incomplete or missing data:
    - Ensure missing requested hours are handled according to existing contract without fabricated values.
    - Case 1: Sparse None values in hourly telemetry are filtered without crashing or substituting 0.
    - Case 2: All values for a metric are None -> metric is not fabricated.
    - Case 3: Requested timeframe not available in forecast -> timeframe_error recorded, no fabricated metrics.
    """
    service = WeatherService()
    mock_data = _generate_mock_open_meteo_response(start_date="2026-10-20")

    # Afternoon hours 12..16: some None values
    mock_data["hourly"]["precipitation"][12] = None
    mock_data["hourly"]["precipitation"][13] = 4.5
    mock_data["hourly"]["precipitation"][14] = None
    mock_data["hourly"]["precipitation"][15] = 2.1
    mock_data["hourly"]["precipitation"][16] = None

    # Entirely None for uv_index
    for h in range(12, 17):
        mock_data["hourly"]["uv_index"][h] = None

    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_data
        mock_get.return_value = mock_resp

        parsed, err = service.fetch_live_weather(
            latitude=22.57, longitude=88.36, timeframe="this afternoon"
        )

        assert err is None
        assert parsed is not None
        # Max of [4.5, 2.1]
        assert parsed["timeframe_precipitation"] == 4.5
        # UV index was entirely None -> must NOT fabricate 0.0 or speculative value
        assert "timeframe_uv_index" not in parsed

    # Case 3: Truncated 12-hour forecast requesting tomorrow afternoon
    short_mock = {
        "latitude": 22.57,
        "longitude": 88.36,
        "current": {"time": "2026-10-20T10:00", "temperature_2m": 25.0},
        "hourly": {
            "time": ["2026-10-20T00:00", "2026-10-20T01:00"],
            "temperature_2m": [20.0, 21.0],
        },
    }
    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = short_mock
        mock_get.return_value = mock_resp

        short_res, _ = service.fetch_live_weather(
            latitude=22.57, longitude=88.36, timeframe="tomorrow afternoon"
        )
        assert short_res is not None
        assert "timeframe_error" in short_res
        assert "No forecast hours available" in short_res["timeframe_error"]
        assert "timeframe_temperature_2m" not in short_res


# =========================================================================
# Test F: Backward compatibility
# =========================================================================
def test_f_backward_compatibility_no_timeframe():
    """
    Test F — Backward compatibility:
    - Verify existing weather calls without a specific timeframe continue to behave as intended.
    """
    service = WeatherService()
    mock_data = _generate_mock_open_meteo_response(start_date="2026-10-20")

    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_data
        mock_get.return_value = mock_resp

        parsed, err = service.fetch_live_weather(
            latitude=22.57, longitude=88.36, location_name="Kolkata", timeframe=None
        )

        assert err is None
        assert parsed is not None
        assert parsed["timeframe_requested"] is None
        assert parsed["temperature_2m"] == 22.0
        assert parsed["precipitation"] == 0.0
        assert parsed["wind_speed_10m"] == 10.0
        assert "timeframe_temperature_2m" not in parsed
        assert "timeframe_precipitation" not in parsed


# =========================================================================
# Test G: Downstream safety & Deterministic Policy Engine Integration
# =========================================================================
def test_g_downstream_safety_policy_engine_integration():
    """
    Test G — Downstream safety:
    Confirm that the corrected telemetry reaches the deterministic policy engine
    and that the policy engine uses the corrected values.

    Scenario:
    - Today's afternoon is mild (0.0 mm rain, 12 km/h wind) -> Policy Engine evaluates LOW risk for picnic.
    - Tomorrow's afternoon is an extreme thunderstorm (25.0 mm rain, 55 km/h wind, code 95) -> Policy Engine evaluates CRITICAL override.
    - Querying "this afternoon" must NOT trigger the severe weather override.
    - Querying "tomorrow afternoon" MUST trigger the severe weather override.
    """
    service = WeatherService()
    engine = DeterministicPolicyEngine()
    context = UserContext(activity="picnic", location="Kolkata", demographics=["all"])

    mock_data = _generate_mock_open_meteo_response(
        start_date="2026-10-20",
        today_afternoon_rain=0.0,
        tomorrow_afternoon_rain=25.0,
        today_afternoon_temp=24.0,
        tomorrow_afternoon_temp=18.0,
    )
    # Set tomorrow's wind to 55 km/h and code to 95 for storm override
    for h in range(36, 41):
        mock_data["hourly"]["wind_speed_10m"][h] = 55.0
        mock_data["hourly"]["weather_code"][h] = 95

    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_data
        mock_get.return_value = mock_resp

        # Step 1: User asks for "this afternoon"
        today_telemetry, _ = service.fetch_live_weather(
            latitude=22.57, longitude=88.36, timeframe="this afternoon"
        )
        today_risk, today_matched = engine.evaluate_risk(
            weather=today_telemetry,
            context=context,
            candidate_sops=[],
            raw_query="Can we have an outdoor picnic this afternoon?",
        )
        # Severe weather override MUST NOT trigger today because today's afternoon is mild!
        assert today_risk.risk_level != "CRITICAL"
        assert today_risk.primary_sop_id != "SOP-OVERRIDE-001"

        # Step 2: User asks for "tomorrow afternoon"
        tom_telemetry, _ = service.fetch_live_weather(
            latitude=22.57, longitude=88.36, timeframe="tomorrow afternoon"
        )
        tom_risk, tom_matched = engine.evaluate_risk(
            weather=tom_telemetry,
            context=context,
            candidate_sops=[],
            raw_query="Can we have an outdoor picnic tomorrow afternoon?",
        )
        # Severe weather override MUST trigger tomorrow because tomorrow's afternoon has 25mm rain and 55km/h wind!
        assert tom_risk.risk_level == "CRITICAL"
        assert tom_risk.primary_sop_id == "SOP-OVERRIDE-001"
