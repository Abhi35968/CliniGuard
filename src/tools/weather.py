from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
import requests
from src.config import (
    OPEN_METEO_GEOCODING_URL,
    OPEN_METEO_FORECAST_URL,
    WEATHER_API_TIMEOUT,
)
from src.tools.base import BaseTool

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


# Common city aliases, historical designations, and spelling variants mapped to official GeoNames indexes
CITY_ALIASES: Dict[str, str] = {
    "bangalore": "Bengaluru",
    "banglore": "Bengaluru",
    "bengaluru": "Bengaluru",
    "bombay": "Mumbai",
    "calcutta": "Kolkata",
    "madras": "Chennai",
    "trivandrum": "Thiruvananthapuram",
    "cochin": "Kochi",
    "pondicherry": "Puducherry",
    "poona": "Pune",
    "baroda": "Vadodara",
    "benares": "Varanasi",
    "banaras": "Varanasi",
    "gurgaon": "Gurugram",
    "allahabad": "Prayagraj",
    "mangalore": "Mangaluru",
    "mysore": "Mysuru",
    "calicut": "Kozhikode",
    "simla": "Shimla",
    "peking": "Beijing",
    "saigon": "Ho Chi Minh City",
    "canton": "Guangzhou",
}


def _parse_geocoding_hit(hit: Dict[str, Any]) -> Dict[str, Any]:
    """Converts a raw Open-Meteo geocoding result into a normalised location dict."""
    admin1 = hit.get("admin1", "")
    country = hit.get("country", "Unknown")
    name = hit.get("name", "")
    display = f"{name}, {admin1}, {country}".strip(", ")
    return {
        "name": name,
        "display_name": display,
        "latitude": float(hit.get("latitude", 0.0)),
        "longitude": float(hit.get("longitude", 0.0)),
        "country": country,
        "admin1": admin1,
        "timezone": hit.get("timezone", "UTC"),
        "population": hit.get("population", 0),
    }


def _parse_timestamp_to_local(
    ts: str, tz: Optional[ZoneInfo] = None
) -> Optional[Tuple[date, int]]:
    """
    Parses an ISO timestamp string into (local_date, local_hour).
    Respects timezone conversions when timestamp includes UTC offset or Z.
    Naive timestamps are treated as already in the forecast location's local time.
    """
    if not ts or not isinstance(ts, str):
        return None
    try:
        clean_ts = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean_ts)
        if dt.tzinfo is not None and tz is not None:
            dt = dt.astimezone(tz)
        return dt.date(), dt.hour
    except (ValueError, TypeError):
        try:
            parts = ts.split("T")
            d_parts = [int(p) for p in parts[0].split("-")]
            d = date(d_parts[0], d_parts[1], d_parts[2])
            h = int(parts[1].split(":")[0]) if len(parts) > 1 else 12
            return d, h
        except Exception:
            return None


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

    # ------------------------------------------------------------------
    # Geocoding
    # ------------------------------------------------------------------

    def resolve_location(
        self, city_name: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Geocodes city name using Open-Meteo search endpoint.
        Returns (best_match_location_dict, error_str).
        The best match is the result with the highest population (most prominent city).
        Use resolve_location_multi() to get all candidates for disambiguation.
        """
        candidates, err = self.resolve_location_multi(city_name)
        if err or not candidates:
            return None, err or f"Could not find any geographic coordinates for '{city_name}'."
        return candidates[0], None

    def resolve_location_multi(
        self, city_name: str, max_results: int = 5
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Returns ALL geocoding candidates for a city name (up to max_results).
        Results are sorted by population descending so the most prominent city
        is always first — this is the automatic selection used by resolve_location().

        Returns (candidates_list, error_str).
        candidates_list is empty on error.
        """
        if not city_name or not city_name.strip():
            return [], "Empty city name provided for location resolution."

        clean_name = city_name.strip()
        alias_name = CITY_ALIASES.get(clean_name.lower())
        search_names = (
            [alias_name, clean_name]
            if alias_name and alias_name.lower() != clean_name.lower()
            else [clean_name]
        )

        all_candidates: List[Dict[str, Any]] = []
        seen_coords = set()
        network_errors = []

        for s_name in search_names:
            try:
                params = {
                    "name": s_name,
                    "count": max_results,
                    "language": "en",
                    "format": "json",
                }
                resp = requests.get(self.geocoding_url, params=params, timeout=self.timeout)

                if resp.status_code == 200:
                    data = resp.json()
                    raw_results = data.get("results") or []
                    for hit in raw_results:
                        cand = _parse_geocoding_hit(hit)
                        coord_key = (round(cand["latitude"], 3), round(cand["longitude"], 3))
                        if coord_key not in seen_coords:
                            seen_coords.add(coord_key)
                            all_candidates.append(cand)
                else:
                    network_errors.append(f"Status code {resp.status_code}")
            except requests.exceptions.Timeout:
                network_errors.append(f"Request timed out while resolving '{s_name}'")
            except requests.exceptions.RequestException as e:
                network_errors.append(str(e))
            except Exception as e:
                network_errors.append(str(e))

        if not all_candidates:
            if network_errors:
                return [], f"Geocoding error: {network_errors[0]}"
            return [], f"Could not find any geographic coordinates for '{clean_name}'."

        # Sort by population descending (most prominent city first)
        all_candidates.sort(key=lambda c: c.get("population", 0) or 0, reverse=True)
        return all_candidates[:max_results], None

    # ------------------------------------------------------------------
    # Weather
    # ------------------------------------------------------------------

    def fetch_live_weather(
        self,
        latitude: float,
        longitude: float,
        location_name: str = "",
        timeframe: Optional[str] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Fetches live forecast and current metrics from Open-Meteo."""
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

            if timeframe and "hourly" in data:
                hourly = data["hourly"]
                target_hour_indices = self._get_timeframe_indices(
                    timestamps=hourly.get("time", []),
                    timeframe=timeframe,
                    reference_time=current.get("time"),
                    timezone_str=data.get("timezone"),
                )
                if target_hour_indices:
                    # Safely extract numeric metrics, skipping missing/None values without fabricating zeroes
                    hourly_temps = [
                        float(hourly["temperature_2m"][i])
                        for i in target_hour_indices
                        if i < len(hourly.get("temperature_2m", []))
                        and hourly["temperature_2m"][i] is not None
                        and isinstance(hourly["temperature_2m"][i], (int, float))
                    ]
                    hourly_rain_prob = [
                        int(hourly["precipitation_probability"][i])
                        for i in target_hour_indices
                        if i < len(hourly.get("precipitation_probability", []))
                        and hourly["precipitation_probability"][i] is not None
                        and isinstance(hourly["precipitation_probability"][i], (int, float))
                    ]
                    hourly_rain = [
                        float(hourly["precipitation"][i])
                        for i in target_hour_indices
                        if i < len(hourly.get("precipitation", []))
                        and hourly["precipitation"][i] is not None
                        and isinstance(hourly["precipitation"][i], (int, float))
                    ]
                    hourly_wind = [
                        float(hourly["wind_speed_10m"][i])
                        for i in target_hour_indices
                        if i < len(hourly.get("wind_speed_10m", []))
                        and hourly["wind_speed_10m"][i] is not None
                        and isinstance(hourly["wind_speed_10m"][i], (int, float))
                    ]
                    hourly_uv = [
                        float(hourly["uv_index"][i])
                        for i in target_hour_indices
                        if i < len(hourly.get("uv_index", []))
                        and hourly["uv_index"][i] is not None
                        and isinstance(hourly["uv_index"][i], (int, float))
                    ]
                    hourly_codes = [
                        int(hourly["weather_code"][i])
                        for i in target_hour_indices
                        if i < len(hourly.get("weather_code", []))
                        and hourly["weather_code"][i] is not None
                        and isinstance(hourly["weather_code"][i], int)
                    ]

                    if hourly_temps:
                        parsed_metrics["timeframe_temperature_2m"] = round(sum(hourly_temps) / len(hourly_temps), 1)
                    if hourly_rain_prob:
                        parsed_metrics["timeframe_precipitation_probability"] = max(hourly_rain_prob)
                    if hourly_rain:
                        parsed_metrics["timeframe_precipitation"] = round(max(hourly_rain), 2)
                    if hourly_wind:
                        parsed_metrics["timeframe_wind_speed_10m"] = round(max(hourly_wind), 1)
                    if hourly_uv:
                        parsed_metrics["timeframe_uv_index"] = round(max(hourly_uv), 1)
                    if hourly_codes:
                        worst_code = max(hourly_codes)
                        parsed_metrics["timeframe_weather_code"] = worst_code
                        parsed_metrics["timeframe_weather_description"] = get_weather_description(worst_code)
                else:
                    parsed_metrics["timeframe_error"] = f"No forecast hours available for timeframe '{timeframe}'."

            return parsed_metrics, None

        except requests.exceptions.Timeout:
            return None, "Open-Meteo forecast request timed out."
        except requests.exceptions.RequestException as e:
            return None, f"Network error during weather forecast fetch: {str(e)}"
        except Exception as e:
            return None, f"Unexpected error during weather forecast fetch: {str(e)}"

    def _get_timeframe_indices(
        self,
        timestamps: list,
        timeframe: str,
        reference_time: Optional[str] = None,
        timezone_str: Optional[str] = None,
    ) -> list:
        """
        Date-aware hourly index selector for forecast aggregation.
        Guarantees that matching hours are selected strictly from the date requested by the user,
        preventing cross-date mixing (e.g. today's afternoon vs tomorrow's afternoon).
        Respects local forecast timezone and timestamp formats.
        """
        if not timestamps or not timeframe or not isinstance(timeframe, str):
            return []

        tf = timeframe.lower().strip()

        # Resolve location timezone
        loc_tz: Optional[ZoneInfo] = None
        if timezone_str and timezone_str != "auto":
            try:
                loc_tz = ZoneInfo(timezone_str)
            except Exception:
                loc_tz = None

        # Resolve reference local date (today relative to the forecast location)
        ref_date: Optional[date] = None
        if reference_time:
            parsed_ref = _parse_timestamp_to_local(reference_time, loc_tz)
            if parsed_ref:
                ref_date = parsed_ref[0]

        if ref_date is None:
            for first_ts in timestamps:
                parsed_first = _parse_timestamp_to_local(first_ts, loc_tz)
                if parsed_first:
                    ref_date = parsed_first[0]
                    break

        if ref_date is None:
            ref_date = datetime.now(loc_tz).date() if loc_tz else datetime.now().date()

        # Determine target date from timeframe
        if "tomorrow" in tf or "next day" in tf:
            target_date = ref_date + timedelta(days=1)
        elif "today" in tf or "this" in tf:
            target_date = ref_date
        else:
            # Default to today for unqualified time-of-day queries
            target_date = ref_date

        # Determine hour criteria
        is_morning = "morning" in tf
        is_afternoon = "afternoon" in tf
        is_evening = "evening" in tf
        is_night = "night" in tf or "tonight" in tf

        indices = []
        for idx, ts in enumerate(timestamps):
            parsed = _parse_timestamp_to_local(ts, loc_tz)
            if not parsed:
                continue
            local_d, local_h = parsed

            # For time of day, verify that local_d matches target_date
            if is_morning:
                if local_d == target_date and 6 <= local_h <= 11:
                    indices.append(idx)
            elif is_afternoon:
                if local_d == target_date and 12 <= local_h <= 16:
                    indices.append(idx)
            elif is_evening:
                if local_d == target_date and 17 <= local_h <= 21:
                    indices.append(idx)
            elif is_night:
                if local_d == target_date and (22 <= local_h or local_h <= 5):
                    indices.append(idx)
            else:
                # Unqualified day query (e.g. "today" or "tomorrow" without part of day)
                if local_d == target_date:
                    indices.append(idx)

        return indices


class WeatherTool(BaseTool):
    """Tool wrapper for fetching live meteorological telemetry and geocoding."""

    name = "weather_tool"
    description = "Fetches live weather telemetry and geocodes city names."

    def __init__(self, service: Optional[WeatherService] = None):
        self.service = service or WeatherService()

    def run(
        self,
        location: str,
        timeframe: Optional[str] = None,
        location_index: int = 0,
    ) -> Dict[str, Any]:
        """
        Geocodes location and fetches live weather forecast telemetry.

        Args:
            location:       City name to resolve.
            timeframe:      Optional timeframe string (e.g. 'evening', 'morning').
            location_index: Index into the disambiguation candidates list.
                            0 = most prominent / auto-selected match (default).
                            Pass >0 to force a specific candidate after the user
                            has picked from the disambiguation list.

        Returns dict with keys:
            success (bool), weather_data (dict|None), error (str|None),
            location_candidates (list) – always populated so the frontend can
            show disambiguation UI when len > 1.
        """
        candidates, loc_err = self.service.resolve_location_multi(location)

        if loc_err or not candidates:
            return {
                "success": False,
                "weather_data": None,
                "error": loc_err or f"Location resolution failed for '{location}'",
                "location_candidates": [],
            }

        # Clamp index to valid range
        idx = max(0, min(location_index, len(candidates) - 1))
        loc_info = candidates[idx]

        weather, w_err = self.service.fetch_live_weather(
            latitude=loc_info["latitude"],
            longitude=loc_info["longitude"],
            location_name=loc_info["display_name"],
            timeframe=timeframe,
        )
        if w_err or not weather:
            return {
                "success": False,
                "weather_data": None,
                "error": w_err or "Weather forecast unavailable",
                "location_candidates": candidates,
            }

        return {
            "success": True,
            "weather_data": weather,
            "error": None,
            "location_candidates": candidates,
            "resolved_location": loc_info,
        }


    def resolve_location(
        self, city_name: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Convenience delegation to underlying service."""
        return self.service.resolve_location(city_name)

    def resolve_location_multi(
        self, city_name: str, max_results: int = 5
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Convenience delegation to underlying service returning candidates list and error."""
        return self.service.resolve_location_multi(city_name, max_results=max_results)
