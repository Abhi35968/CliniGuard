from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, START, END

from src.state import AgentState
from src.db import get_sqlite_checkpointer, save_turn
from src.nodes import (
    input_guardrail_node,
    parse_and_extract_node,
    planner_agent_node,
    check_location_condition,
    ask_clarification_node,
    fetch_weather_node,
    check_weather_condition,
    weather_error_node,
    evaluate_sops_node,
    check_sop_condition,
    no_sop_fallback_node,
    generate_advisory_node,
    validate_grounding_node,
    check_grounding_condition,
    deterministic_fallback_node,
)


def build_weather_graph() -> StateGraph:
    """Constructs the LangGraph state machine with input guardrail, planner agent, tools, and output validation."""
    workflow = StateGraph(AgentState)

    # 1. Register Nodes
    workflow.add_node("input_guardrail", input_guardrail_node)
    workflow.add_node("parse_and_extract", parse_and_extract_node)
    workflow.add_node("planner_agent", planner_agent_node)
    workflow.add_node("ask_clarification", ask_clarification_node)
    workflow.add_node("fetch_weather", fetch_weather_node)
    workflow.add_node("weather_error", weather_error_node)
    workflow.add_node("evaluate_sops", evaluate_sops_node)
    workflow.add_node("no_sop_fallback", no_sop_fallback_node)
    workflow.add_node("generate_advisory", generate_advisory_node)
    workflow.add_node("validate_grounding", validate_grounding_node)
    workflow.add_node("deterministic_fallback", deterministic_fallback_node)

    # 2. Add Edges & Conditional Branches
    workflow.add_edge(START, "input_guardrail")

    # Branch 0: Input Guardrail Check
    def check_input_guardrail_condition(state: AgentState) -> str:
        if not state.get("input_guardrail_passed", True):
            return "input_guardrail_failed"
        return "parse_and_extract"

    workflow.add_conditional_edges(
        "input_guardrail",
        check_input_guardrail_condition,
        {
            "input_guardrail_failed": END,
            "parse_and_extract": "parse_and_extract",
        },
    )

    workflow.add_edge("parse_and_extract", "planner_agent")

    # Branch 1: Location & Planning Check
    workflow.add_conditional_edges(
        "planner_agent",
        check_location_condition,
        {
            "ask_clarification": "ask_clarification",
            "fetch_weather": "fetch_weather",
            "input_guardrail_failed": END,
        },
    )

    workflow.add_edge("ask_clarification", END)

    # Branch 2: Weather Fetch Check
    workflow.add_conditional_edges(
        "fetch_weather",
        check_weather_condition,
        {
            "weather_error": "weather_error",
            "evaluate_sops": "evaluate_sops",
        },
    )

    workflow.add_edge("weather_error", END)

    # Branch 3: SOP Match Check
    workflow.add_conditional_edges(
        "evaluate_sops",
        check_sop_condition,
        {
            "no_sop_fallback": "no_sop_fallback",
            "generate_advisory": "generate_advisory",
        },
    )

    workflow.add_edge("no_sop_fallback", "validate_grounding")
    workflow.add_edge("generate_advisory", "validate_grounding")

    # Branch 4: Grounding Guardrail Validation (True Branch with Retry Loop & Fallback)
    workflow.add_conditional_edges(
        "validate_grounding",
        check_grounding_condition,
        {
            "passed": END,
            "retry": "generate_advisory",
            "fallback": "deterministic_fallback",
        },
    )

    workflow.add_edge("deterministic_fallback", END)

    return workflow


# SQLite checkpointer for multi-turn persistent session storage across restarts
sqlite_checkpointer = get_sqlite_checkpointer()
compiled_graph = build_weather_graph().compile(checkpointer=sqlite_checkpointer)


def run_agent_turn(
    user_query: str,
    session_id: str = "default-session",
    thread_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Executes a single conversational turn through the compiled LangGraph workflow.
    Retains state across turns in the same session_id using SQLite persistence.
    """
    config = thread_config or {"configurable": {"thread_id": session_id}}

    # Fetch previous state if available
    current_state_snapshot = compiled_graph.get_state(config)
    prev_values = current_state_snapshot.values if current_state_snapshot else {}

    input_payload = {
        "session_id": session_id,
        "raw_query": user_query,
        "messages": [{"role": "user", "content": user_query}],
        "user_context": prev_values.get("user_context"),
        "location": prev_values.get("location"),
        "activity": prev_values.get("activity"),
        "timeframe": prev_values.get("timeframe"),
        "demographics": prev_values.get("demographics", []),
        "intent": prev_values.get("intent"),
        "pending_clarification": prev_values.get("pending_clarification", False),
        "clarification_target": prev_values.get("clarification_target"),
        "weather_data": prev_values.get("weather_data"),
        "weather_error": None,
        "retrieved_sops": [],
        "matched_sops": [],
        "active_sop": None,
        "risk_assessment": None,
        "sop_citation": None,
        "input_guardrail_passed": None,
        "grounding_valid": None,
        "grounding_errors": [],
        "guardrail_results": [],
        "retry_count": 0,
        "final_response": None,
        "route": None,
        "execution_trace": [],
    }

    result = compiled_graph.invoke(input_payload, config=config)

    # Persist the turn to SQLite database
    try:
        final_text = result.get("final_response", "")
        save_turn(
            session_id=session_id,
            user_query=user_query,
            assistant_response=final_text,
            route=result.get("route"),
            citation=result.get("sop_citation"),
            weather_data=result.get("weather_data"),
            trace=result.get("execution_trace"),
            location=result.get("location"),
            activity=result.get("activity"),
            matched_sops=result.get("matched_sops"),
            active_sop=result.get("active_sop"),
        )
    except Exception as e:
        print(f"[run_agent_turn] Warning: could not log turn to SQLite DB: {e}")

    return result
