import json
import os
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from src.config import SOPS_FILE_PATH

# Severity ranking hierarchy
SEVERITY_RANKS = {
    "CRITICAL": 4,
    "HIGH": 3,
    "MODERATE": 2,
    "LOW": 1,
}

# Synonym mapping for robust intent & activity matching
ACTIVITY_SYNONYMS = {
    "cycling": ["cycling", "cycle", "bike", "biking", "bicycle", "pedal", "two-wheeler", "scooter", "motorcycle", "ride"],
    "running": ["running", "run", "jogging", "jog", "marathon", "sprint", "cardio", "outdoor exercise"],
    "walking": ["walking", "walk", "pedestrian", "stroll", "errands", "foot"],
    "picnic": ["picnic", "park outing", "barbecue", "bbq", "outdoor lunch", "garden party", "family outing", "sitting outside"],
    "park": ["park", "playground", "swings", "slides", "outdoor play", "stroller walk", "kids outdoor"],
    "hiking": ["hiking", "hike", "trekking", "trek", "camping", "trail"],
    "two-wheeler": ["two-wheeler", "scooter", "motorcycle", "bike commute", "scooty", "bike ride", "daily commute"],
    "travel": ["travel", "driving", "drive", "road trip", "highway driving", "long drive", "commute"],
    "dog walking": ["dog walking", "dog walk", "pet walk", "walk my dog", "puppy", "canine", "taking pet out"],
    "morning walk": ["morning walk", "evening walk", "leisure walk", "temple visit", "gardening"],
    "outdoor exercise": ["outdoor exercise", "exercise", "workout", "sports", "football", "cricket", "outdoor training", "training"],
    "swimming": ["swimming", "swim", "pool"],
}

DEMOGRAPHIC_SYNONYMS = {
    "children": ["child", "children", "kid", "kids", "toddler", "toddlers", "baby", "babies", "infant", "infants", "daughter", "son"],
    "elderly": ["elderly", "senior", "seniors", "grandparents", "grandfather", "grandmother", "dada", "dadi", "nana", "nani", "cardiac", "heart patient"],
    "pets": ["pet", "pets", "dog", "dogs", "puppy", "pup", "cat", "animal", "canine"],
    "athletes": ["athlete", "athletes", "runner", "cyclist", "training"],
}


# Indoor activities that have no outdoor safety policy
INDOOR_ACTIVITIES = ["chess", "board game", "cards", "indoor", "living room", "painting indoors", "reading books", "video games"]


class SOPEngine:
    """Decoupled Standard Operating Procedure (SOP) Rule & Evaluation Engine."""

    def __init__(self, sops_path: str = SOPS_FILE_PATH):
        self.sops_path = sops_path
        self._sops_cache: Optional[List[Dict[str, Any]]] = None
        self._last_mtime: float = 0.0

    def load_sops(self, force_reload: bool = False) -> List[Dict[str, Any]]:
        """Loads SOP definitions from JSON file. Auto-reloads if modified."""
        if not os.path.exists(self.sops_path):
            return []

        try:
            mtime = os.path.getmtime(self.sops_path)
            if force_reload or self._sops_cache is None or mtime > self._last_mtime:
                with open(self.sops_path, "r", encoding="utf-8") as f:
                    self._sops_cache = json.load(f)
                self._last_mtime = mtime
            return self._sops_cache or []
        except Exception as e:
            print(f"[SOPEngine] Error loading SOPs: {e}")
            return self._sops_cache or []

    def evaluate(
        self,
        weather: Dict[str, Any],
        activity: Optional[str] = None,
        demographics: Optional[List[str]] = None,
        timeframe: Optional[str] = None,
        raw_query: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Evaluates all active SOPs against weather telemetry and user context.
        Returns list of matched SOPs sorted by severity (highest first).
        """
        sops = self.load_sops()
        matched: List[Dict[str, Any]] = []

        query_lower = (raw_query or "").lower()
        activity_lower = (activity or "").lower()
        demographics_lower = [d.lower() for d in (demographics or [])]

        # If user is asking exclusively about an indoor activity, no outdoor SOP applies
        is_explicit_indoor = any(ia in activity_lower or ia in query_lower for ia in INDOOR_ACTIVITIES)
        has_outdoor_mention = any(oa in query_lower for oa in ["cycling", "bike", "running", "jog", "picnic", "park", "hiking", "travel", "driving", "two-wheeler", "playground", "swim", "outside", "outdoor"])
        if is_explicit_indoor and not has_outdoor_mention:
            return []

        for sop in sops:
            is_match, reason = self._evaluate_single_sop(
                sop=sop,
                weather=weather,
                activity_str=activity_lower,
                demographics=demographics_lower,
                timeframe=timeframe,
                query_str=query_lower,
            )
            if is_match:
                matched_sop = dict(sop)
                matched_sop["match_reason"] = reason
                matched_sop["severity_rank"] = SEVERITY_RANKS.get(sop.get("severity", "LOW"), 1)
                matched.append(matched_sop)

        # Sort by severity rank descending (CRITICAL -> HIGH -> MODERATE -> LOW)
        matched.sort(key=lambda s: s["severity_rank"], reverse=True)
        return matched

    def _evaluate_single_sop(
        self,
        sop: Dict[str, Any],
        weather: Dict[str, Any],
        activity_str: str,
        demographics: List[str],
        timeframe: Optional[str],
        query_str: str,
    ) -> Tuple[bool, str]:
        """Evaluates whether a single SOP matches given context and weather."""
        sop_id = sop.get("id", "UNKNOWN")
        category = sop.get("category", "")
        target_acts = sop.get("target_activities", [])
        target_demos = sop.get("target_demographics", [])

        # 1. Severe Weather Overrides match all activities
        is_override = category == "severe_override" or "*" in target_acts

        # 2. Activity & Demographic Relevance Check
        if not is_override:
            act_match = self._matches_activity(target_acts, activity_str, query_str)
            if not act_match:
                return False, f"Activity mismatch for {sop_id}"

            # If SOP specifically targets a specialized demographic (e.g. kids, elderly, pets), demographic match is required
            if "all" not in target_demos:
                demo_match = self._matches_demographics(target_demos, demographics, query_str)
                if not demo_match:
                    return False, f"Demographic mismatch for {sop_id}"

        # 3. Time Window Check (if specified in SOP)
        time_window = sop.get("time_window")
        if time_window:
            if not self._check_time_window(time_window, timeframe):
                return False, f"Outside time window {time_window} for {sop_id}"

        # 4. Condition Evaluation
        conditions = sop.get("conditions", {})
        cond_type = conditions.get("type", "numeric")

        if cond_type == "fuzzy_comfort":
            is_valid, reason = self._evaluate_fuzzy_comfort(conditions.get("evaluation", {}), weather)
            if not is_valid:
                return False, reason
            return True, f"Fuzzy comfort criteria satisfied: {reason}"

        elif cond_type in ("numeric", "composite", "categorical"):
            is_valid, reason = self._evaluate_condition_tree(conditions, weather)
            if not is_valid:
                return False, reason
            return True, f"Weather criteria met: {reason}"

        return False, "Unknown condition structure"

    def _matches_activity(self, target_acts: List[str], activity_str: str, query_str: str) -> bool:
        """Checks if target activities match extracted activity or query keywords."""
        if "*" in target_acts:
            return True

        act_clean = activity_str.strip() if activity_str else ""
        query_clean = query_str.strip() if query_str else ""

        for target in target_acts:
            t_low = target.lower().strip()
            if not t_low:
                continue

            # Exact or substring match on extracted activity
            if act_clean and (t_low in act_clean or act_clean in t_low):
                return True
            # Word-level match in query string
            if re.search(r"\b" + re.escape(t_low) + r"\b", query_clean):
                return True
            # Check synonyms
            syns = ACTIVITY_SYNONYMS.get(t_low, [])
            for syn in syns:
                if act_clean and (syn in act_clean or act_clean in syn):
                    return True
                if re.search(r"\b" + re.escape(syn) + r"\b", query_clean):
                    return True
        return False

    def _matches_demographics(self, target_demos: List[str], demographics: List[str], query_str: str) -> bool:
        """Checks if target demographics match user context."""
        if "all" in target_demos:
            return True

        query_clean = query_str.strip() if query_str else ""
        clean_demos = [d.lower().strip() for d in demographics if d and d.strip()]

        for target in target_demos:
            t_low = target.lower().strip()
            if not t_low:
                continue

            if any(t_low in d for d in clean_demos):
                return True
            if re.search(r"\b" + re.escape(t_low) + r"\b", query_clean):
                return True
            syns = DEMOGRAPHIC_SYNONYMS.get(t_low, [])
            for syn in syns:
                if any(syn in d for d in clean_demos) or re.search(r"\b" + re.escape(syn) + r"\b", query_clean):
                    return True
        return False

    def _check_time_window(self, time_window: Dict[str, int], timeframe: Optional[str]) -> bool:
        """Verifies if current or requested timeframe overlaps with SOP time window."""
        start_h = time_window.get("start_hour", 0)
        end_h = time_window.get("end_hour", 24)

        if timeframe:
            tf = timeframe.lower()
            # If asking specifically about morning, afternoon, evening
            if "afternoon" in tf and (start_h <= 14 <= end_h):
                return True
            if "midday" in tf and (start_h <= 12 <= end_h):
                return True
            if "morning" in tf and (start_h <= 9 <= end_h):
                return True
            if "evening" in tf and (start_h <= 18 <= end_h):
                return True
            if "today" in tf or "now" in tf:
                current_hour = datetime.now().hour
                return start_h <= current_hour <= end_h

        # Default to checking current hour
        current_hour = datetime.now().hour
        return start_h <= current_hour <= end_h

    def _evaluate_condition_tree(self, cond_node: Dict[str, Any], weather: Dict[str, Any]) -> Tuple[bool, str]:
        """Recursively evaluates composite or leaf rule conditions."""
        node_type = cond_node.get("type", "numeric")
        operator = cond_node.get("operator", "AND").upper()
        rules = cond_node.get("rules", [])

        if not rules:
            return True, "No rules"

        eval_results = []
        reasons = []

        for r in rules:
            if "rules" in r or r.get("type") == "composite":
                sub_res, sub_reason = self._evaluate_condition_tree(r, weather)
                eval_results.append(sub_res)
                reasons.append(sub_reason)
            else:
                leaf_res, leaf_reason = self._evaluate_leaf_rule(r, weather)
                eval_results.append(leaf_res)
                reasons.append(leaf_reason)

        if operator == "AND":
            passed = all(eval_results)
            return passed, " & ".join(reasons) if passed else f"AND condition failed: {[r for r, ok in zip(reasons, eval_results) if not ok]}"
        elif operator == "OR":
            passed = any(eval_results)
            return passed, " | ".join([r for r, ok in zip(reasons, eval_results) if ok]) if passed else "No OR condition met"

        return False, "Unknown operator"

    def _evaluate_leaf_rule(self, rule: Dict[str, Any], weather: Dict[str, Any]) -> Tuple[bool, str]:
        """Evaluates single atomic condition like (temperature_2m >= 35.0)."""
        field = rule.get("field", "")
        op = rule.get("op", "==")
        target_val = rule.get("value")

        # Allow using timeframe-adjusted values if present and relevant
        actual_val = weather.get(f"timeframe_{field}", weather.get(field))

        if actual_val is None:
            return False, f"Missing weather telemetry for field '{field}'"

        try:
            if op == ">=":
                res = float(actual_val) >= float(target_val)
                return res, f"{field} ({actual_val}) >= {target_val}"
            elif op == "<=":
                res = float(actual_val) <= float(target_val)
                return res, f"{field} ({actual_val}) <= {target_val}"
            elif op == ">":
                res = float(actual_val) > float(target_val)
                return res, f"{field} ({actual_val}) > {target_val}"
            elif op == "<":
                res = float(actual_val) < float(target_val)
                return res, f"{field} ({actual_val}) < {target_val}"
            elif op == "==":
                res = actual_val == target_val
                return res, f"{field} ({actual_val}) == {target_val}"
            elif op == "!=":
                res = actual_val != target_val
                return res, f"{field} ({actual_val}) != {target_val}"
            elif op == "in":
                res = actual_val in target_val
                return res, f"{field} ({actual_val}) in {target_val}"
            else:
                return False, f"Unsupported operator '{op}'"
        except Exception as e:
            return False, f"Error evaluating {field} {op} {target_val}: {e}"

    def _evaluate_fuzzy_comfort(self, bounds: Dict[str, Any], weather: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Evaluates fuzzy composite recreational/picnic comfort index.
        Non-binary composite evaluation across temperature, rain, wind, and UV.
        """
        temp = float(weather.get("temperature_2m", 20.0))
        rain_prob = int(weather.get("precipitation_probability", 0))
        rain = float(weather.get("precipitation", 0.0))
        wind = float(weather.get("wind_speed_10m", 0.0))
        uv = float(weather.get("uv_index", 0.0))

        min_t = bounds.get("ideal_temp_min", 18.0)
        max_t = bounds.get("ideal_temp_max", 28.0)
        max_rp = bounds.get("max_rain_probability", 30)
        max_r = bounds.get("max_rain_amount", 0.5)
        max_w = bounds.get("max_wind_speed", 22.0)
        max_u = bounds.get("max_uv_index", 7.0)

        penalties = []
        if temp < min_t:
            penalties.append(f"Cool temperature ({temp}°C < {min_t}°C)")
        elif temp > max_t:
            penalties.append(f"Warm temperature ({temp}°C > {max_t}°C)")

        if rain_prob > max_rp:
            penalties.append(f"Elevated rain probability ({rain_prob}% > {max_rp}%)")
        if rain > max_r:
            penalties.append(f"Active precipitation ({rain} mm > {max_r} mm)")
        if wind > max_w:
            penalties.append(f"Breezy winds ({wind} km/h > {max_w} km/h)")
        if uv > max_u:
            penalties.append(f"High UV radiation ({uv} > {max_u})")

        # Picnic comfort passes if no major environmental violations
        if len(penalties) == 0:
            return True, f"Optimal picnic conditions: {temp}°C, {wind} km/h wind, {rain_prob}% rain chance, UV {uv}."
        elif len(penalties) == 1:
            # Tolerable with mild note
            return True, f"Fair picnic conditions with minor caution: {penalties[0]}"
        else:
            return False, f"Unfavorable picnic conditions due to: {', '.join(penalties)}"
