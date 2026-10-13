import pytest
from src.weather import WeatherService, get_weather_description


def test_geocoding_success():
    service = WeatherService()
    loc, err = service.resolve_location("Bhopal")
    assert err is None
    assert loc is not None
    assert "Bhopal" in loc["name"]
    assert isinstance(loc["latitude"], float)
    assert isinstance(loc["longitude"], float)


def test_geocoding_invalid_city():
    service = WeatherService()
    loc, err = service.resolve_location("NonExistentCityNameXYZ9999")
    assert loc is None
    assert err is not None
    assert "Could not find any geographic coordinates" in err


def test_fetch_live_weather_success():
    service = WeatherService()
    # Test coordinates for Delhi
    data, err = service.fetch_live_weather(28.6139, 77.2090, location_name="Delhi")
    assert err is None
    assert data is not None
    assert "temperature_2m" in data
    assert "wind_speed_10m" in data
    assert "precipitation" in data
    assert "precipitation_probability" in data
    assert "uv_index" in data


def test_fetch_live_weather_with_timeframe():
    service = WeatherService()
    data, err = service.fetch_live_weather(
        12.9716, 77.5946, location_name="Bengaluru", timeframe="evening"
    )
    assert err is None
    assert data is not None
    assert data["timeframe_requested"] == "evening"


def test_weather_description_mapping():
    assert get_weather_description(0) == "Clear sky"
    assert get_weather_description(95) == "Thunderstorm (slight or moderate)"
    assert get_weather_description(None) == "Unknown"
