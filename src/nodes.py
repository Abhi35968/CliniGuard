import json
import re
import time
from typing import Dict, Any, List, Optional

from src.state import AgentState
from src.schemas.context import UserContext
from src.schemas.risk import RiskAssessment
from src.tools.weather import WeatherTool, WeatherService
from src.tools.sop_tool import SOPRetrieverTool
from src.agents.context_agent import ContextExtractionAgent
from src.agents.planner import PlannerAgent
from src.policy.engine import DeterministicPolicyEngine
from src.guardrails.input import InputGuardrail
from src.guardrails.output import OutputGuardrailPipeline
from src.llm.factory import get_llm_for_task
from src.llm.prompts import ADVISORY_GENERATION_SYSTEM_PROMPT
from src.observability.tracer import ExecutionTracer

# Singleton component instances
weather_tool = WeatherTool()
weather_service = weather_tool.service
context_agent = ContextExtractionAgent()
planner_agent = PlannerAgent()
sop_retriever_tool = SOPRetrieverTool()
policy_engine = DeterministicPolicyEngine()
input_guardrail = InputGuardrail()
output_guardrail_pipeline = OutputGuardrailPipeline()


def input_guardrail_node(state: AgentState) -> Dict[str, Any]:
    """Validates user query against prompt injection and security policies."""
    query = state.get("raw_query", "").strip()
    result = input_guardrail.validate_input(query)

    if not result.passed:
        msg = (
            f"### 🛡️ CliniGuard Security & Compliance Notice\n\n"
            f"Your request could not be processed because it violates system security guidelines.\n\n"
            f"- **Security Details:** {result.details}\n"
            f"- **Notice:** System safety policies and clinical SOPs cannot be bypassed or overridden.\n\n"
            f"---\n"
            f"*Status: Refused by Input Security Guardrail.*"
        )
        return {
            "input_guardrail_passed": False,
            "final_response": msg,
            "sop_citation": "INPUT_GUARDRAIL_BLOCKED",
            "route": "input_guardrail_failed",
            "guardrail_results": [result.model_dump()],
            "execution_trace": [f"Input Guardrail: FAILED ({result.details})"],
            "messages": [{"role": "assistant", "content": msg}],
        }

    return {
        "input_guardrail_passed": True,
        "guardrail_results": [result.model_dump()],
        "execution_trace": ["Input Guardrail: PASSED."],
    }


def parse_and_extract_node(state: AgentState) -> Dict[str, Any]:
    """
    Parses user query and merges extracted entities with session history using ContextExtractionAgent.
    Handles mid-clarification state if bot previously requested missing info.
    """
    query = state.get("raw_query", "").strip()
    existing_loc = state.get("location")
    existing_act = state.get("activity")
    existing_tf = state.get("timeframe")
    existing_demos = state.get("demographics", []) or []
    is_mid_clarification = state.get("pending_clarification", False)
    clarification_target = state.get("clarification_target")

    # Case A: Handling pending clarification (e.g. user supplied city name directly)
    if is_mid_clarification and clarification_target == "location":
        extracted_cand = context_agent.extract_context(query, {})
        new_loc = extracted_cand.location or query.replace("in ", "").replace("at ", "").strip().title()
        
        trace_entry = f"Mid-clarification resolved: Location='{new_loc}' (Preserved Activity='{existing_act}')"
        return {
            "user_context": UserContext(
                location=new_loc,
                activity=existing_act,
                timeframe=existing_tf or "today",
                demographics=existing_demos,
            ).model_dump(),
            "location": new_loc,
            "activity": existing_act,
            "timeframe": existing_tf or "today",
            "demographics": existing_demos,
            "pending_clarification": False,
            "clarification_target": None,
            "execution_trace": [trace_entry],
        }

    # Case B: Standard extraction via ContextExtractionAgent
    session_ctx = {
        "location": existing_loc,
        "activity": existing_act,
        "timeframe": existing_tf,
        "demographics": existing_demos,
    }
    extracted_ctx = context_agent.extract_context(query, session_ctx)

    trace_entry = (
        f"Context Agent Extracted: Loc='{extracted_ctx.location}', Act='{extracted_ctx.activity}', "
        f"TF='{extracted_ctx.timeframe}', Demo={extracted_ctx.demographics}"
    )

    return {
        "user_context": extracted_ctx.model_dump(),
        "location": extracted_ctx.location,
        "activity": extracted_ctx.activity,
        "timeframe": extracted_ctx.timeframe,
        "demographics": extracted_ctx.demographics,
        "pending_clarification": False,
        "clarification_target": None,
        "execution_trace": [trace_entry],
    }


def planner_agent_node(state: AgentState) -> Dict[str, Any]:
    """Executes PlannerAgent to determine tool execution flow."""
    ctx_dict = state.get("user_context") or {}
    ctx = UserContext(**ctx_dict) if ctx_dict else UserContext(
        location=state.get("location"),
        activity=state.get("activity"),
        timeframe=state.get("timeframe"),
        demographics=state.get("demographics", []),
    )
    query = state.get("raw_query", "")

    plan = planner_agent.plan(ctx, query)
    trace_msg = f"Planner Agent Decision: needs_clarification={plan.needs_clarification}, tools={plan.tools_to_run}, reasoning='{plan.reasoning}'"

    return {
        "pending_clarification": plan.needs_clarification,
        "clarification_target": plan.clarification_target,
        "execution_trace": [trace_msg],
    }


def check_location_condition(state: AgentState) -> str:
    """Conditional routing based on location availability."""
    if not state.get("input_guardrail_passed", True):
        return "input_guardrail_failed"
    if state.get("pending_clarification") or not state.get("location"):
        return "ask_clarification"
    return "fetch_weather"


def ask_clarification_node(state: AgentState) -> Dict[str, Any]:
    """Generates formal clarification prompt when location is missing."""
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
    """Fetches live meteorological telemetry using WeatherTool."""
    location = state.get("location", "")
    timeframe = state.get("timeframe")

    res = weather_tool.run(location=location, timeframe=timeframe)

    if not res["success"] or not res["weather_data"]:
        err_msg = res.get("error") or f"Could not find coordinates or weather for '{location}'"
        return {
            "weather_data": None,
            "weather_error": err_msg,
            "execution_trace": [f"Weather Tool error: {err_msg}"],
        }

    weather = res["weather_data"]
    trace_msg = (
        f"Weather Tool fetched for {weather['location_name']}: "
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
    """Retrieves SOPs via SOPRetrieverTool and evaluates RiskAssessment via DeterministicPolicyEngine."""
    weather = state.get("weather_data", {})
    query = state.get("raw_query", "")

    ctx_dict = state.get("user_context") or {}
    ctx = UserContext(**ctx_dict) if ctx_dict else UserContext(
        location=state.get("location"),
        activity=state.get("activity"),
        timeframe=state.get("timeframe"),
        demographics=state.get("demographics", []),
    )

    # 1. Retrieve candidates via SOPRetrieverTool
    ret_res = sop_retriever_tool.run(query=query, context=ctx, top_k=5)
    candidate_sops = ret_res.get("retrieved_sops", [])

    # 2. Evaluate risk via DeterministicPolicyEngine
    risk_assessment, matched = policy_engine.evaluate_risk(
        weather=weather,
        context=ctx,
        candidate_sops=candidate_sops,
        raw_query=query,
    )

    if not matched:
        return {
            "retrieved_sops": candidate_sops,
            "matched_sops": [],
            "active_sop": None,
            "risk_assessment": risk_assessment.model_dump(),
            "sop_citation": "NO_SOP_APPLICABLE",
            "execution_trace": [f"Retrieved {len(candidate_sops)} SOP candidates -> 0 matched policy conditions."],
        }

    primary = matched[0]
    citation_str = risk_assessment.sop_citation

    trace_msg = (
        f"Retrieved {len(candidate_sops)} SOPs -> Evaluated {len(matched)} matched -> "
        f"Primary: {primary['id']} [Severity: {primary['severity']}, Priority: {primary.get('priority', 0)}]"
    )

    return {
        "retrieved_sops": candidate_sops,
        "matched_sops": matched,
        "active_sop": primary,
        "risk_assessment": risk_assessment.model_dump(),
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
    """Helper to format structured clinical markdown advisory grounded in SOP and telemetry."""
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
    """Generates grounded advisory using LLM or structured template grounded strictly in RiskAssessment & telemetry."""
    w = state.get("weather_data", {})
    matched = state.get("matched_sops", [])
    primary_sop = state.get("active_sop", matched[0] if matched else {})
    timeframe = state.get("timeframe") or "current"
    loc = w.get("location_name") or state.get("location") or "the requested location"
    act = state.get("activity") or "outdoor activity"
    retry_count = state.get("retry_count", 0)

    # 1. Try grounded LLM generation
    llm_provider = get_llm_for_task("generation")
    response_text = None

    if llm_provider and llm_provider.chat_model:
        try:
            risk_dict = state.get("risk_assessment") or {}
            user_msg = (
                f"User Context -> Activity: {act}, Location: {loc}, Timeframe: {timeframe}\n"
                f"Verified Telemetry -> Temp: {w.get('temperature_2m')}°C, Rain: {w.get('precipitation')}mm ({w.get('precipitation_probability')}%), Wind: {w.get('wind_speed_10m')}km/h, UV: {w.get('uv_index')}, Conditions: {w.get('weather_description')}\n"
                f"RiskAssessment -> {json.dumps(risk_dict)}\n"
                f"Primary SOP -> ID: {primary_sop.get('id')}, Title: '{primary_sop.get('title')}', Severity: {primary_sop.get('severity')}\n"
                f"Guidance -> {primary_sop.get('guidance')}\n"
                f"Precautions -> {json.dumps(primary_sop.get('precautions', []))}"
            )
            response_text = llm_provider.generate(
                system_prompt=ADVISORY_GENERATION_SYSTEM_PROMPT,
                user_message=user_msg,
            )
        except Exception as e:
            print(f"[generate_advisory_node] LLM generation error: {e}. Using golden clinical template.")

    # 2. Fallback to 100% verified clinical markdown advisory template
    if not response_text or len(response_text.strip()) < 30 or primary_sop.get("id") not in response_text:
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
        "execution_trace": [f"Generated advisory (attempt={retry_count}) citing {primary_sop.get('id')}"],
    }


def validate_grounding_node(state: AgentState) -> Dict[str, Any]:
    """Executes OutputGuardrailPipeline verifying grounding, numeric consistency, and policy alignment."""
    response = state.get("final_response", "")
    primary = state.get("active_sop")
    matched = state.get("matched_sops", [])
    weather = state.get("weather_data") or {}
    risk_dict = state.get("risk_assessment")
    risk_assessment = RiskAssessment(**risk_dict) if risk_dict else None
    retry_count = state.get("retry_count", 0)

    overall_passed, check_results = output_guardrail_pipeline.validate_output(
        response_text=response,
        risk_assessment=risk_assessment,
        matched_sops=matched if matched else ([primary] if primary else []),
        telemetry=weather,
    )

    errors = []
    for r in check_results:
        if not r.passed:
            errors.extend(r.violations)

    if overall_passed:
        return {
            "grounding_valid": True,
            "grounding_errors": [],
            "guardrail_results": [c.model_dump() for c in check_results],
            "execution_trace": ["Guardrail Check: PASSED (Strictly grounded in SOP & API numbers)."],
            "messages": [{"role": "assistant", "content": response}],
        }
    else:
        new_retry = retry_count + 1
        return {
            "grounding_valid": False,
            "grounding_errors": errors,
            "guardrail_results": [c.model_dump() for c in check_results],
            "retry_count": new_retry,
            "execution_trace": [f"Guardrail Check: FAILED ({errors}), incrementing retry_count={new_retry}"],
        }


def check_grounding_condition(state: AgentState) -> str:
    """Conditional routing for guardrail validation."""
    if state.get("grounding_valid", True):
        return "passed"
    elif state.get("retry_count", 0) < 2:
        return "retry"
    else:
        return "fallback"


def deterministic_fallback_node(state: AgentState) -> Dict[str, Any]:
    """Deterministic safety fallback: delivers 100% verified golden clinical template when LLM fails guardrails."""
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
