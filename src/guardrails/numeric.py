import re
from typing import Dict, Any, List
from src.schemas.trace import GuardrailCheckResult


class NumericConsistencyGuardrail:
    """Verifies that numerical facts mentioned in response text strictly match verified telemetry."""

    def validate_numeric_consistency(
        self, response_text: str, telemetry: Dict[str, Any]
    ) -> GuardrailCheckResult:
        violations: List[str] = []

        if not response_text or not telemetry:
            return GuardrailCheckResult(
                guardrail_type="numeric",
                passed=True,
                details="No response text or telemetry to evaluate.",
                violations=[],
            )

        # Extract temperature claims e.g. "34°C" or "34.0°C" or "34 degrees"
        temp_matches = re.findall(r"(\d+(?:\.\d+)?)\s*°?C", response_text)
        actual_temp = telemetry.get("temperature_2m")

        if actual_temp is not None and temp_matches:
            found_actual = any(abs(float(t_str) - float(actual_temp)) < 0.5 for t_str in temp_matches)
            if not found_actual:
                violations.append(
                    f"Temperature numbers in response {temp_matches} do not match verified telemetry ({actual_temp}°C)."
                )

        # Extract wind claims e.g. "45 km/h" or "45.0 km/h"
        wind_matches = re.findall(r"(\d+(?:\.\d+)?)\s*km/h", response_text)
        actual_wind = telemetry.get("wind_speed_10m")

        if actual_wind is not None and wind_matches:
            found_actual = any(abs(float(w_str) - float(actual_wind)) < 0.5 for w_str in wind_matches)
            if not found_actual:
                violations.append(
                    f"Wind speed numbers in response {wind_matches} do not match verified telemetry ({actual_wind} km/h)."
                )

        passed = len(violations) == 0
        details = "Numeric consistency check PASSED." if passed else f"Numeric consistency check FAILED: {', '.join(violations)}"

        return GuardrailCheckResult(
            guardrail_type="numeric",
            passed=passed,
            details=details,
            violations=violations,
        )
