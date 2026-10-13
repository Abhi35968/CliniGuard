from typing import Dict, Any, Optional, Tuple
import requests
from src.config import (
    OPEN_METEO_GEOCODING_URL,
    OPEN_METEO_FORECAST_URL,
    WEATHER_API_TIMEOUT,
)

# WMO Weather interpretation codes
WMO_WEATHER_CODES: Dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense intensity drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    62: "Moderate rain",
    63: "Heavy rain",
    65: "Heavy continuous rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm (slight or moderate)",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def get_weather_description(code: Optional[int]) -> str:
    """Returns human-readable text for WMO weather code."""
    if code is None:
        return "Unknown"
    return WMO_WEATHER_CODES.get(code, f"Weather condition code {code}")


class WeatherService:
    """Client for Open-Meteo geocoding and live weather telemetry."""

    def __init__(
        self,
        geocoding_url: str = OPEN_METEO_GEOCODING_URL,
        forecast_url: str = OPEN_METEO_FORECAST_URL,
        timeout: float = WEATHER_API_TIMEOUT,
    ):
        self.geocoding_url = geocoding_url
        self.forecast_url = forecast_url
        self.timeout = timeout

    def resolve_location(
        self, city_name: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Geocodes city name using Open-Meteo search endpoint.
        Returns (location_dict, error_message).
        """
        if not city_name or not city_name.strip():
            return None, "Empty city name provided for location resolution."

        clean_name = city_name.strip()
        try:
            params = {"name": clean_name, "count": 5, "language": "en", "format": "json"}
            resp = requests.get(self.geocoding_url, params=params, timeout=self.timeout)
            
            if resp.status_code != 200:
                return None, f"Geocoding service returned status code {resp.status_code}."
            
            data = resp.json()
            results = data.get("results")
            if not results or len(results) == 0:
                return None, f"Could not find any geographic coordinates for '{clean_name}'."
            
            first_hit = results[0]
            return {
                "name": first_hit.get("name"),
                "latitude": float(first_hit.get("latitude")),
                "longitude": float(first_hit.get("longitude")),
                "country": first_hit.get("country", "Unknown"),
                "admin1": first_hit.get("admin1", ""),
                "timezone": first_hit.get("timezone", "UTC"),
            }, None

        except requests.exceptions.Timeout:
            return None, f"Geocoding request timed out while resolving '{clean_name}'."
        except requests.exceptions.RequestException as e:
            return None, f"Network error during geocoding: {str(e)}"
        except Exception as e:
            return None, f"Unexpected error during geocoding: {str(e)}"

    def fetch_live_weather(
        self,
        latitude: float,
        longitude: float,
        location_name: str = "",
        timeframe: Optional[str] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Fetches live forecast and current metrics from Open-Meteo.
        Returns (weather_data_dict, error_message).
        """
        try:
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "current": [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "apparent_temperature",
                    "precipitation",
                    "precipitation_probability",
                    "weather_code",
                    "cloud_cover",
                    "wind_speed_10m",
                    "wind_gusts_10m",
                    "uv_index",
                ],
                "hourly": [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "apparent_temperature",
                    "precipitation_probability",
                    "precipitation",
                    "weather_code",
                    "wind_speed_10m",
                    "uv_index",
                ],
                "forecast_days": 2,
                "timezone": "auto",
            }

            resp = requests.get(self.forecast_url, params=params, timeout=self.timeout)
            
            if resp.status_code != 200:
                return None, f"Weather forecast API returned status code {resp.status_code}."
            
            data = resp.json()
            current = data.get("current", {})
            if not current:
                return None, "Open-Meteo response did not contain current weather values."

            weather_code = current.get("weather_code")
            weather_desc = get_weather_description(weather_code)

            # Standardized verified metrics
            parsed_metrics = {
                "location_name": location_name,
                "latitude": latitude,
                "longitude": longitude,
                "time": current.get("time"),
                "temperature_2m": float(current.get("temperature_2m", 0.0)),
                "apparent_temperature": float(current.get("apparent_temperature", current.get("temperature_2m", 0.0))),
                "relative_humidity_2m": int(current.get("relative_humidity_2m", 0)),
                "precipitation": float(current.get("precipitation", 0.0)),
                "precipitation_probability": int(current.get("precipitation_probability", 0)),
                "wind_speed_10m": float(current.get("wind_speed_10m", 0.0)),
                "wind_gusts_10m": float(current.get("wind_gusts_10m", current.get("wind_speed_10m", 0.0))),
                "uv_index": float(current.get("uv_index", 0.0)),
                "cloud_cover": int(current.get("cloud_cover", 0)),
                "weather_code": weather_code,
                "weather_description": weather_desc,
                "raw_response": data,
                "timeframe_requested": timeframe,
            }

            # If user asked for a specific timeframe (e.g. "evening", "tomorrow morning"), adjust relevant metrics
            if timeframe and "hourly" in data:
                hourly = data["hourly"]
                target_hour_indices = self._get_timeframe_indices(hourly.get("time", []), timeframe)
                if target_hour_indices:
                    # Compute average or representative metrics for that timeframe
                    hourly_temps = [hourly["temperature_2m"][i] for i in target_hour_indices if i < len(hourly["temperature_2m"])]
                    hourly_rain_prob = [hourly["precipitation_probability"][i] for i in target_hour_indices if i < len(hourly["precipitation_probability"])]
                    hourly_rain = [hourly["precipitation"][i] for i in target_hour_indices if i < len(hourly["precipitation"])]
                    hourly_wind = [hourly["wind_speed_10m"][i] for i in target_hour_indices if i < len(hourly["wind_speed_10m"])]
                    hourly_uv = [hourly["uv_index"][i] for i in target_hour_indices if i < len(hourly["uv_index"])]
                    hourly_codes = [hourly["weather_code"][i] for i in target_hour_indices if i < len(hourly["weather_code"])]

                    if hourly_temps:
                        parsed_metrics["timeframe_temperature_2m"] = round(sum(hourly_temps) / len(hourly_temps), 1)
                    if hourly_rain_prob:
                        parsed_metrics["timeframe_precipitation_probability"] = max(hourly_rain_prob)
                    if hourly_rain:
                        parsed_metrics["timeframe_precipitation"] = max(hourly_rain)
                    if hourly_wind:
                        parsed_metrics["timeframe_wind_speed_10m"] = round(max(hourly_wind), 1)
                    if hourly_uv:
                        parsed_metrics["timeframe_uv_index"] = round(max(hourly_uv), 1)
                    if hourly_codes:
                        worst_code = max(hourly_codes)
                        parsed_metrics["timeframe_weather_code"] = worst_code
                        parsed_metrics["timeframe_weather_description"] = get_weather_description(worst_code)

            return parsed_metrics, None

        except requests.exceptions.Timeout:
            return None, "Open-Meteo forecast request timed out."
        except requests.exceptions.RequestException as e:
            return None, f"Network error during weather forecast fetch: {str(e)}"
        except Exception as e:
            return None, f"Unexpected error during weather forecast fetch: {str(e)}"

    def _get_timeframe_indices(self, timestamps: list, timeframe: str) -> list:
        """Finds indices in hourly timestamps corresponding to timeframe keyword."""
        tf = timeframe.lower()
        indices = []
        for idx, ts in enumerate(timestamps[:48]):  # first 48 hours
            # ts format "2026-09-04T18:00"
            hour = int(ts.split("T")[1].split(":")[0]) if "T" in ts else 12
            if "evening" in tf and 17 <= hour <= 21:
                indices.append(idx)
            elif "morning" in tf and 6 <= hour <= 11:
                indices.append(idx)
            elif "afternoon" in tf and 12 <= hour <= 16:
                indices.append(idx)
            elif "night" in tf and (22 <= hour or hour <= 5):
                indices.append(idx)
        return indices
