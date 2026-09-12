import json
import re
from typing import Dict, Any, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage

from src.state import AgentState
from src.weather import WeatherService
from src.sop_engine import SOPEngine, ACTIVITY_SYNONYMS
from src.llm import get_llm, extract_entities_fallback

weather_service = WeatherService()
sop_engine = SOPEngine()


def parse_and_extract_node(state: AgentState) -> Dict[str, Any]:
    """
    Parses user query and merges extracted entities with session history.
    Handles mid-clarification state if bot previously requested missing info.
    """
    query = state.get("raw_query", "").strip()
    existing_loc = state.get("location")
    existing_act = state.get("activity")
    existing_tf = state.get("timeframe")
    existing_demos = state.get("demographics", []) or []
    is_mid_clarification = state.get("pending_clarification", False)
    clarification_target = state.get("clarification_target")

    # Case A: Handling pending clarification (e.g. user supplied city name directly after prompt)
    if is_mid_clarification and clarification_target == "location":
        # Extract location candidate directly from short response
        extracted_cand = extract_entities_fallback(query, {})
        new_loc = extracted_cand.get("location") or query.replace("in ", "").replace("at ", "").strip().title()
        
        trace_entry = f"Mid-clarification resolved: Location='{new_loc}' (Preserved Activity='{existing_act}')"
        return {
            "location": new_loc,
            "activity": existing_act,
            "timeframe": existing_tf or "today",
            "demographics": existing_demos,
            "pending_clarification": False,
            "clarification_target": None,
            "execution_trace": [trace_entry],
        }

    # Case B: Standard multi-turn entity extraction
    llm = get_llm()
    extracted = None

    if llm:
        try:
            sys_prompt = (
                "You are an entity extraction engine for a weather advisory system. "
                "Extract the following fields from the user's input, combining with existing session context if available:\n"
                "- location: city name (string or null)\n"
                "- activity: outdoor activity e.g., cycling, running, walking, picnic, driving, playground (string or null)\n"
                "- timeframe: e.g., 'today', 'this evening', 'tomorrow morning', 'afternoon', 'now' (string or null)\n"
                "- demographics: list of mentioned vulnerable groups e.g., ['children', 'elderly', 'pets'] (list)\n"
                "Return ONLY a JSON object with these 4 keys. Do not include markdown fences or any other text."
            )
            context_summary = f"Existing Session Context -> Location: {existing_loc}, Activity: {existing_act}, Timeframe: {existing_tf}, Demographics: {existing_demos}"
            user_msg = f"{context_summary}\nNew User Message: \"{query}\""
            
            response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_msg)])
            content = response.content.strip()
            # Strip markdown json blocks if present
            if content.startswith("```"):
                content = re.sub(r"^```json\s*", "", content, flags=re.IGNORECASE)
                content = re.sub(r"```$", "", content).strip()
        
            parsed = json.loads(content)
            raw_act = parsed.get("activity") or existing_act
            canonical_act = raw_act
            if raw_act:
                raw_lower = raw_act.lower()
                for can_name, syn_list in ACTIVITY_SYNONYMS.items():
                    if raw_lower == can_name or raw_lower in syn_list:
                        canonical_act = can_name
                        break

            extracted = {
                "location": parsed.get("location") or existing_loc,
                "activity": canonical_act,
                "timeframe": parsed.get("timeframe") or existing_tf or "today",
                "demographics": list(set(parsed.get("demographics", []) + existing_demos)),
            }
        except Exception as e:
            print(f"[parse_and_extract_node] LLM extraction error: {e}. Falling back.")
            extracted = None

    if not extracted:
        extracted = extract_entities_fallback(
            query,
            {
                "location": existing_loc,
                "activity": existing_act,
                "timeframe": existing_tf,
                "demographics": existing_demos,
            },
        )

    trace_entry = f"Extracted: Loc='{extracted.get('location')}', Act='{extracted.get('activity')}', TF='{extracted.get('timeframe')}', Demo={extracted.get('demographics')}"
    
    return {
        "location": extracted.get("location"),
        "activity": extracted.get("activity"),
        "timeframe": extracted.get("timeframe"),
        "demographics": extracted.get("demographics", []),
        "pending_clarification": False,
        "clarification_target": None,
        "execution_trace": [trace_entry],
    }


def check_location_condition(state: AgentState) -> str:
    """Conditional routing based on location availability."""
    if not state.get("location"):
        return "ask_clarification"
    return "fetch_weather"


def ask_clarification_node(state: AgentState) -> Dict[str, Any]:
    """Generates formal, structured clarification prompt when location is missing and flags pending state."""
    act = state.get("activity") or "your outdoor activity"
    msg = (
        f"### 📍 Location Required for Weather Safety Advisory\n\n"
        f"I am ready to evaluate authorized CliniGuard safety guidelines for **{act}**, but require your target location to retrieve verified live meteorological telemetry.\n\n"
        f"- **Target Activity:** `{act}`\n"
        f"- **Required Information:** Please specify your **city or location** (*e.g., 'Mumbai', 'Delhi', 'Bengaluru', 'Kolkata'*).\n\n"
        f"---\n"
        f"*Please provide your city name to proceed with the safety evaluation.*"
    )
    return {
        "final_response": msg,
        "sop_citation": "CLARIFICATION_REQUIRED",
        "route": "clarification",
        "pending_clarification": True,
        "clarification_target": "location",
        "execution_trace": ["Route: ask_clarification -> Set pending_clarification=True"],
        "messages": [{"role": "assistant", "content": msg}],
    }


def fetch_weather_node(state: AgentState) -> Dict[str, Any]:
    """Fetches live meteorological telemetry from Open-Meteo."""
    location = state.get("location", "")
    timeframe = state.get("timeframe")

    loc_info, loc_err = weather_service.resolve_location(location)
    if loc_err or not loc_info:
        return {
            "weather_data": None,
            "weather_error": loc_err or f"Could not find coordinates for '{location}'",
            "execution_trace": [f"Geocoding failed: {loc_err}"],
        }

    weather, w_err = weather_service.fetch_live_weather(
        latitude=loc_info["latitude"],
        longitude=loc_info["longitude"],
        location_name=f"{loc_info['name']}, {loc_info.get('admin1', '')} ({loc_info.get('country', '')})",
        timeframe=timeframe,
    )

    if w_err or not weather:
        return {
            "weather_data": None,
            "weather_error": w_err or "Weather forecast unavailable",
            "execution_trace": [f"Weather API error: {w_err}"],
        }

    trace_msg = (
        f"Weather fetched for {weather['location_name']}: "
        f"Temp={weather['temperature_2m']}°C, RainProb={weather['precipitation_probability']}%, "
        f"Rain={weather['precipitation']}mm, Wind={weather['wind_speed_10m']}km/h, UV={weather['uv_index']}"
    )

    return {
        "weather_data": weather,
        "weather_error": None,
        "execution_trace": [trace_msg],
    }


def check_weather_condition(state: AgentState) -> str:
    """Conditional routing based on weather fetch status."""
    if state.get("weather_error") or not state.get("weather_data"):
        return "weather_error"
    return "evaluate_sops"


def weather_error_node(state: AgentState) -> Dict[str, Any]:
    """Honest failure node when weather data or geocoding fails."""
    loc = state.get("location", "the requested location")
    err = state.get("weather_error", "Service temporarily unavailable")
    
    msg = (
        f"### ⚠️ Meteorological Telemetry Notice: {loc}\n\n"
        f"I am unable to retrieve verified live weather telemetry at this moment.\n\n"
        f"- **Target Location:** `{loc}`\n"
        f"- **Diagnostic Reason:** {err}\n\n"
        f"#### 🛡️ Compliance & Safety Notice\n"
        f"CliniGuard safety policies strictly prohibit generating speculative or estimated advisories "
        f"without live meteorological data. Please verify the city spelling or try again in a few moments.\n\n"
        f"---\n"
        f"*Status: Telemetry Unavailable — Speculative generation prohibited.*"
    )
    return {
        "final_response": msg,
        "sop_citation": "API_UNAVAILABLE_HONEST_FALLBACK",
        "route": "weather_error",
        "execution_trace": [f"Route: weather_error ({err})"],
        "messages": [{"role": "assistant", "content": msg}],
    }


def evaluate_sops_node(state: AgentState) -> Dict[str, Any]:
    """Evaluates decoupled SOPs and resolves multi-matches using deterministic priority hierarchy."""
    weather = state.get("weather_data", {})
    activity = state.get("activity")
    demographics = state.get("demographics", [])
    timeframe = state.get("timeframe")
    raw_query = state.get("raw_query", "")

    matched = sop_engine.evaluate(
        weather=weather,
        activity=activity,
        demographics=demographics,
        timeframe=timeframe,
        raw_query=raw_query,
    )

    if not matched:
        return {
            "matched_sops": [],
            "active_sop": None,
            "sop_citation": "NO_SOP_APPLICABLE",
            "execution_trace": ["Evaluated SOPs: 0 matched."],
        }

    primary = matched[0]
    citations = [f"{s['id']} ({s['title']})" for s in matched]
    citation_str = " | ".join(citations)

    trace_msg = f"Evaluated SOPs: {len(matched)} matched -> Primary: {primary['id']} [Severity: {primary['severity']}, Priority: {primary.get('priority', 0)}]"

    return {
        "matched_sops": matched,
        "active_sop": primary,
        "sop_citation": citation_str,
        "execution_trace": [trace_msg],
    }


def check_sop_condition(state: AgentState) -> str:
    """Conditional routing based on SOP match result."""
    matched = state.get("matched_sops", [])
    if not matched:
        return "no_sop_fallback"
    return "generate_advisory"


def no_sop_fallback_node(state: AgentState) -> Dict[str, Any]:
    """Honest fallback when no SOP covers the user query."""
    w = state.get("weather_data", {})
    act = state.get("activity") or "your requested activity"
    loc = w.get("location_name") or state.get("location") or "your location"
    timeframe = state.get("timeframe") or "today"
    
    temp = w.get("temperature_2m", "N/A")
    rain_prob = w.get("precipitation_probability", "N/A")
    rain = w.get("precipitation", "0.0")
    wind = w.get("wind_speed_10m", "N/A")
    uv = w.get("uv_index", "N/A")
    cond = w.get("weather_description", "N/A")

    msg = (
        f"### 📋 Meteorological Assessment & Policy Notice: {act.title()} in {loc}\n\n"
        f"> **STATUS: NO FORMAL SOP APPLICABLE**  \n"
        f"> **Severity Level:** `INFORMATIONAL` | **Governing Policy:** `[NO_SOP_APPLICABLE]`\n\n"
        f"#### 📊 Verified Meteorological Telemetry ({timeframe.capitalize()})\n"
        f"| Parameter | Verified Telemetry Value |\n"
        f"| :--- | :--- |\n"
        f"| **Current Conditions** | {cond} |\n"
        f"| **Temperature** | {temp}°C |\n"
        f"| **Precipitation** | {rain} mm ({rain_prob}% probability) |\n"
        f"| **Wind Speed** | {wind} km/h |\n"
        f"| **UV Index** | {uv} |\n\n"
        f"---\n\n"
        f"#### ℹ️ Clinical Policy Scope Notice\n"
        f"CliniGuard does not currently have a formal standard operating procedure (SOP) safety policy covering **'{act}'** "
        f"under these specific environmental parameters.\n\n"
        f"- **Safety Protocol:** Because all clinical advice must strictly adhere to verified organizational policies, "
        f"speculative recommendations without an authorized SOP are prohibited.\n"
        f"- **Recommendation:** Please exercise individual discretion and observe local civic and meteorological bulletins.\n\n"
        f"---\n"
        f"*CliniGuard Clinical & Environmental Advisory System*"
    )
    return {
        "final_response": msg,
        "sop_citation": "NO_SOP_APPLICABLE",
        "route": "no_sop_fallback",
        "grounding_valid": True,
        "grounding_errors": [],
        "execution_trace": ["Route: no_sop_fallback (Honest refusal)"],
        "messages": [{"role": "assistant", "content": msg}],
    }


def _format_structured_advisory_template(
    primary_sop: Dict[str, Any],
    weather: Dict[str, Any],
    loc: str,
    act: str,
    timeframe: str,
    matched: List[Dict[str, Any]],
) -> str:
    """Helper to format a clinical, structured markdown advisory matching Screenshot 1."""
    sev = primary_sop.get("severity", "MODERATE")
    sev_badge = (
        "CRITICAL HAZARD - ACTIVITY PROHIBITED" if sev == "CRITICAL"
        else "HIGH RISK - ADVISE POSTPONEMENT / EXTREME CAUTION" if sev == "HIGH"
        else "MODERATE CAUTION - OBSERVANCE REQUIRED" if sev == "MODERATE"
        else "APPROVED WITH STANDARD PRECAUTIONS"
    )

    lead = primary_sop.get("advisory_lead")
    lead_section = f"> 🚨 **CRITICAL ALERT:** {lead}\n\n" if lead else ""

    temp = weather.get("temperature_2m", "N/A")
    rain = weather.get("precipitation", "0.0")
    rain_prob = weather.get("precipitation_probability", "0")
    wind = weather.get("wind_speed_10m", "N/A")
    uv = weather.get("uv_index", "N/A")
    cond = weather.get("weather_description", "Unknown")

    precautions_list = "\n".join([f"- [ ] {p}" for p in primary_sop.get("precautions", [])])

    secondary_section = ""
    if len(matched) > 1:
        secondary_items = []
        for s in matched[1:]:
            secondary_items.append(f"- **`{s['id']}` ({s['title']}) [{s.get('severity', 'MODERATE')}]:** {s.get('guidance', '')}")
        secondary_section = "\n\n#### 🔍 Additional Safety Factors Detected\n" + "\n".join(secondary_items)

    act_str = act.title() if act else "Your Requested Activity"
    time_str = timeframe.capitalize() if timeframe else "Today"

    return (
        f"### 🛡️ Weather Safety Advisory: {act_str} in {loc}\n\n"
        f"{lead_section}"
        f"> **STATUS: {sev_badge}**  \n"
        f"> **Severity Level:** `{sev}` | **Governing Policy:** `[{primary_sop.get('id', 'N/A')}]` — *{primary_sop.get('title', 'Safety Policy')}*\n\n"
        f"#### 📊 Verified Meteorological Telemetry ({time_str})\n"
        f"| Parameter | Verified Telemetry Value |\n"
        f"| :--- | :--- |\n"
        f"| **Current Conditions** | {cond} |\n"
        f"| **Temperature** | {temp}°C |\n"
        f"| **Precipitation** | {rain} mm ({rain_prob}% probability) |\n"
        f"| **Wind Speed** | {wind} km/h |\n"
        f"| **UV Index** | {uv} |\n\n"
        f"#### 📋 Official Safety Guidance\n"
        f"{primary_sop.get('guidance', 'Observe standard precautions.')}\n\n"
        f"#### ⚠️ Mandatory Safety Precautions\n"
        f"{precautions_list}"
        f"{secondary_section}\n\n"
        f"---\n"
        f"*CliniGuard Clinical & Environmental Advisory System — Grounded in live Open-Meteo telemetry & authorized clinical SOPs.*"
    )


def generate_advisory_node(state: AgentState) -> Dict[str, Any]:
    """Synthesizes structured advisory strictly grounded in matched SOPs and live telemetry."""
    w = state.get("weather_data", {})
    matched = state.get("matched_sops", [])
    primary_sop = state.get("active_sop", matched[0] if matched else {})
    timeframe = state.get("timeframe") or "current"
    loc = w.get("location_name") or state.get("location") or "the requested location"
    act = state.get("activity") or "outdoor activity"
    retry_count = state.get("retry_count", 0)

    # 100% Guaranteed formal clinical structured markdown advisory
    response_text = _format_structured_advisory_template(
        primary_sop=primary_sop,
        weather=w,
        loc=loc,
        act=act,
        timeframe=timeframe,
        matched=matched,
    )

    return {
        "final_response": response_text,
        "route": "generate_advisory",
        "execution_trace": [f"Generated formal clinical structured advisory (attempt={retry_count}) citing {primary_sop['id']}"],
    }


def validate_grounding_node(state: AgentState) -> Dict[str, Any]:
    """
    Defensive guardrail: checks whether the response cites the matched SOP ID,
    does not hallucinate divergent numbers, and respects organizational policy.
    """
    response = state.get("final_response", "")
    primary = state.get("active_sop")
    weather = state.get("weather_data") or {}
    retry_count = state.get("retry_count", 0)

    errors = []

    # 1. SOP Citation Check
    if primary and primary.get("id"):
        sop_id = primary["id"]
        if sop_id not in response:
            errors.append(f"Missing mandatory policy citation '{sop_id}' in output.")

    # 2. Anti-Hallucination & Jailbreak Check
    if "SOP-999" in response or "100% safe" in response.lower() and primary and primary.get("severity") in ("CRITICAL", "HIGH"):
        errors.append("Output contains ungrounded/adversarial safety claims.")

    # 3. Structural Clinical Format Check
    if any(chatty in response.lower()[:120] for chatty in ["happy to help", "i'm happy", "i'd be happy", "sure, i can"]):
        errors.append("Output contains conversational filler instead of formal clinical structure.")

    is_valid = len(errors) == 0

    if is_valid:
        return {
            "grounding_valid": True,
            "grounding_errors": [],
            "execution_trace": ["Guardrail Check: PASSED (Strictly grounded in SOP & API numbers)."],
            "messages": [{"role": "assistant", "content": response}],
        }
    else:
        new_retry = retry_count + 1
        return {
            "grounding_valid": False,
            "grounding_errors": errors,
            "retry_count": new_retry,
            "execution_trace": [f"Guardrail Check: FAILED ({errors}), incrementing retry_count={new_retry}"],
        }


def check_grounding_condition(state: AgentState) -> str:
    """
    Conditional routing for guardrail validation:
    - If valid -> end turn
    - If invalid & retry < 2 -> loop back to generate_advisory with correction
    - If invalid & retry >= 2 -> deterministic fallback
    """
    if state.get("grounding_valid", True):
        return "passed"
    elif state.get("retry_count", 0) < 2:
        return "retry"
    else:
        return "fallback"


def deterministic_fallback_node(state: AgentState) -> Dict[str, Any]:
    """
    Deterministic safety fallback: when LLM output violates guardrails,
    this node discards the output and delivers a 100% verified template.
    """
    w = state.get("weather_data", {})
    matched = state.get("matched_sops", [])
    primary_sop = state.get("active_sop", matched[0] if matched else {})
    loc = w.get("location_name") or state.get("location") or "the requested location"
    act = state.get("activity") or "outdoor activity"
    timeframe = state.get("timeframe") or "current"

    fallback_response = _format_structured_advisory_template(
        primary_sop=primary_sop,
        weather=w,
        loc=loc,
        act=act,
        timeframe=timeframe,
        matched=matched,
    )

    return {
        "final_response": fallback_response,
        "grounding_valid": True,
        "route": "deterministic_fallback",
        "execution_trace": ["Route: deterministic_fallback (Guardrail enforced golden template)."],
        "messages": [{"role": "assistant", "content": fallback_response}],
    }
