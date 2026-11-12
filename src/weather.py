"""
Legacy module re-exporting WeatherService, WeatherTool, and helpers from src.tools.weather.
Preserves backwards compatibility with existing imports.
"""

from src.tools.weather import (
    WMO_WEATHER_CODES,
    get_weather_description,
    WeatherService,
    WeatherTool,
)

__all__ = [
    "WMO_WEATHER_CODES",
    "get_weather_description",
    "WeatherService",
    "WeatherTool",
]
