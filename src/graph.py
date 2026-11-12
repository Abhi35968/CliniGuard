import threading
from contextlib import contextmanager
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, START, END

from src.state import AgentState
from src.db import get_sqlite_checkpointer, save_turn
from src.observability import get_logger, SessionLogContext

logger = get_logger("graph")

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

# Module-level lock protecting compiled_graph and sqlite_checkpointer reassignments
_graph_lock = threading.RLock()


class _TurnCoordinator:
    """
    Coordinates active agent turns with administrative global operations (clear_all_sessions).
    Allows unlimited concurrent turns across distinct sessions while serializing global clears.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._turn_cond = threading.Condition(self._lock)
        self._active_turns = 0
        self._exclusive_active = False

    @contextmanager
    def active_turn(self):
        with self._turn_cond:
            while self._exclusive_active:
                self._turn_cond.wait()
            self._active_turns += 1
        try:
            yield
        finally:
            with self._turn_cond:
                self._active_turns -= 1
                if self._active_turns == 0:
                    self._turn_cond.notify_all()

    @contextmanager
    def exclusive(self):
        with self._turn_cond:
            while self._exclusive_active or self._active_turns > 0:
                self._turn_cond.wait()
            self._exclusive_active = True
        try:
            yield
        finally:
            with self._turn_cond:
                self._exclusive_active = False
                self._turn_cond.notify_all()


turn_coordinator = _TurnCoordinator()


class _SessionLockEntry:
    __slots__ = ("lock", "ref_count")

    def __init__(self):
        self.lock = threading.Lock()
        self.ref_count = 0


# Per-session lock registry to serialize multi-turn state transitions on the SAME session_id
# while allowing distinct sessions to execute concurrently in parallel without contention.
_session_locks: Dict[str, _SessionLockEntry] = {}
_session_locks_guard = threading.Lock()


@contextmanager
def session_scope_lock(session_id: str):
    """
    Context manager acquiring a thread lock for the given session_id.
    Reference-counted so inactive locks are reclaimed safely without race conditions.
    """
    with _session_locks_guard:
        if session_id not in _session_locks:
            _session_locks[session_id] = _SessionLockEntry()
        entry = _session_locks[session_id]
        entry.ref_count += 1

    try:
        with entry.lock:
            yield entry.lock
    finally:
        with _session_locks_guard:
            entry.ref_count -= 1
            # Safely reclaim inactive locks when no threads are active or waiting and registry exceeds 500
            if len(_session_locks) > 500 and entry.ref_count <= 0:
                to_remove = [k for k, v in _session_locks.items() if v.ref_count <= 0]
                for k in to_remove:
                    _session_locks.pop(k, None)


def _get_session_lock(session_id: str):
    """Backward-compatibility helper returning session_scope_lock."""
    return session_scope_lock(session_id)


def recompile_graph(checkpointer=None):
    """Recompiles the graph workflow atomically under _graph_lock."""
    global sqlite_checkpointer, compiled_graph
    with _graph_lock:
        sqlite_checkpointer = checkpointer or get_sqlite_checkpointer()
        compiled_graph = build_weather_graph().compile(checkpointer=sqlite_checkpointer)
        return compiled_graph


def run_agent_turn(
    user_query: str,
    session_id: str = "default-session",
    thread_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Executes a single conversational turn through the compiled LangGraph workflow.
    Retains state across turns in the same session_id using SQLite persistence.
    Synchronized per session_id to guarantee turn ordering and checkpoint consistency.
    """
    with SessionLogContext(session_id):
        with turn_coordinator.active_turn():
            with session_scope_lock(session_id):
                config = thread_config or {"configurable": {"thread_id": session_id}}

            # Bind to active compiled graph atomically under _graph_lock
            with _graph_lock:
                active_graph = compiled_graph

        # Fetch previous state if available
        current_state_snapshot = active_graph.get_state(config)
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

        result = active_graph.invoke(input_payload, config=config)

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
                retrieved_sops=result.get("retrieved_sops"),
                retrieval_error=result.get("retrieval_error"),
            )
        except Exception as e:
            logger.warning("Could not log turn to SQLite DB for session %s: %s", session_id, e)

        return result


# Human-friendly stage descriptions for real-time progress streaming
NODE_STAGE_MESSAGES = {
    "input_guardrail": "🛡️ Scanning query for adversarial injections and security policies...",
    "parse_and_extract": "🧠 Analyzing intent: extracting location, activity, and demographics...",
    "planner_agent": "🎯 Validating query parameters and planning execution path...",
    "ask_clarification": "📍 Location missing; requesting clarification from user...",
    "fetch_weather": "🌤️ Fetching verified meteorological telemetry from Open-Meteo...",
    "weather_error": "⚠️ Live weather telemetry is currently unavailable...",
    "evaluate_sops": "📋 Evaluating clinical SOP policies against verified telemetry...",
    "no_sop_fallback": "ℹ️ Applying general standard precautions for current conditions...",
    "generate_advisory": "🤖 Synthesizing evidence-grounded safety advisory...",
    "validate_grounding": "🔍 Output Guardrail: Checking numerical consistency and SOP grounding...",
    "deterministic_fallback": "⚠️ Safety trigger: Deploying 100% deterministic clinical fallback...",
}


def stream_agent_turn(
    user_query: str,
    session_id: str = "default-session",
    thread_config: Optional[Dict[str, Any]] = None,
):
    """
    Generator executing the LangGraph workflow in streaming mode.
    Yields structured event dicts:
      - {"event": "status", "data": {"stage": str, "message": str}}
      - {"event": "provisional", "data": {"text": str, "is_verified": False}}
      - {"event": "result", "data": QueryResponse dict}
      - {"event": "done", "data": {"status": "completed"}}
      - {"event": "error", "data": {"error": str}}
    """
    with SessionLogContext(session_id):
        with turn_coordinator.active_turn():
            with session_scope_lock(session_id):
                config = thread_config or {"configurable": {"thread_id": session_id}}

            # Bind to active compiled graph atomically under _graph_lock
            with _graph_lock:
                active_graph = compiled_graph

        # Fetch previous state from checkpointer
        try:
            current_state_snapshot = active_graph.get_state(config)
            prev_values = current_state_snapshot.values if current_state_snapshot else {}
        except Exception:
            prev_values = {}

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

        last_emitted_text = None

        try:
            for chunk in active_graph.stream(input_payload, config=config, stream_mode="updates"):
                for node_name, node_output in chunk.items():
                    msg = NODE_STAGE_MESSAGES.get(node_name, f"Executing {node_name}...")
                    yield {
                        "event": "status",
                        "data": {
                            "stage": node_name,
                            "message": msg,
                        },
                    }

                    # If the advisory generation node produced provisional text, stream it with clear unverified label
                    if node_name == "generate_advisory" and "final_response" in node_output:
                        prov_text = node_output.get("final_response", "")
                        if prov_text and prov_text != last_emitted_text:
                            last_emitted_text = prov_text
                            yield {
                                "event": "provisional",
                                "data": {
                                    "text": prov_text,
                                    "is_verified": False,
                                    "warning": "Unvalidated draft; awaiting output grounding check.",
                                },
                            }

        except Exception as e:
            yield {
                "event": "error",
                "data": {"error": f"Execution failed: {str(e)}"},
            }
            return

        # Obtain final state from compiled graph checkpointer
        try:
            final_snapshot = active_graph.get_state(config)
            final_state = final_snapshot.values if final_snapshot else {}
        except Exception:
            final_state = {}

        user_context = final_state.get("user_context") or {}
        risk_assessment = final_state.get("risk_assessment") or {}
        loc_candidates = final_state.get("location_candidates") or []

        final_advisory = final_state.get("final_response") or "No advisory produced."
        is_safe = risk_assessment.get("is_safe", True) if isinstance(risk_assessment, dict) else getattr(risk_assessment, "is_safe", True)
        risk_level = risk_assessment.get("risk_level") if isinstance(risk_assessment, dict) else getattr(risk_assessment, "risk_level", None)
        resolved_loc = (
            user_context.get("location")
            if isinstance(user_context, dict)
            else getattr(user_context, "location", None)
        ) or final_state.get("location")

        # Persist the turn to SQLite
        try:
            save_turn(
                session_id=session_id,
                user_query=user_query,
                assistant_response=final_advisory,
                route=final_state.get("route"),
                citation=final_state.get("sop_citation"),
                weather_data=final_state.get("weather_data"),
                trace=final_state.get("execution_trace"),
                location=final_state.get("location"),
                activity=final_state.get("activity"),
                matched_sops=final_state.get("matched_sops"),
                active_sop=final_state.get("active_sop"),
                retrieved_sops=final_state.get("retrieved_sops"),
                retrieval_error=final_state.get("retrieval_error"),
            )
        except Exception as e:
            logger.warning("Could not log stream turn to SQLite DB for session %s: %s", session_id, e)

        # Emit the authoritative validated result event
        yield {
            "event": "result",
            "data": {
                "thread_id": session_id,
                "advisory": final_advisory,
                "final_response": final_advisory,
                "is_safe": is_safe,
                "risk_level": risk_level,
                "risk_assessment": risk_assessment,
                "location": resolved_loc,
                "location_candidates": loc_candidates,
                "weather": final_state.get("weather_data"),
                "weather_data": final_state.get("weather_data"),
                "matched_sops": final_state.get("matched_sops") or [],
                "active_sop": final_state.get("active_sop"),
                "retrieved_sops": final_state.get("retrieved_sops") or [],
                "retrieval_error": final_state.get("retrieval_error"),
                "sop_citation": final_state.get("sop_citation"),
                "route": final_state.get("route"),
                "execution_trace": final_state.get("execution_trace") or [],
                "guardrail_results": final_state.get("guardrail_results") or [],
                "input_guardrail_passed": final_state.get("input_guardrail_passed", True),
                "requires_clarification": final_state.get("pending_clarification", False),
                "clarification_question": (
                    final_advisory if final_state.get("pending_clarification") else None
                ),
                "grounding_valid": final_state.get("grounding_valid", True),
                "is_verified": True,
            },
        }

        # Emit completion event
        yield {
            "event": "done",
            "data": {"status": "completed"},
        }
