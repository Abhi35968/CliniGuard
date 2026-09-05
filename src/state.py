from typing import Dict, Any, List, Optional, Annotated
from typing_extensions import TypedDict
import operator


class AgentState(TypedDict):
    """LangGraph state representation for Weather-Advisory Assistant."""

    # Session & Chat history
    session_id: str
    messages: Annotated[List[Dict[str, Any]], operator.add]
    raw_query: str

    # Context & Entities (carried across turns in session)
    location: Optional[str]
    activity: Optional[str]
    timeframe: Optional[str]
    demographics: List[str]
    intent: Optional[str]

    # Mid-clarification state flags
    pending_clarification: Optional[bool]
    clarification_target: Optional[str]

    # Weather Telemetry
    weather_data: Optional[Dict[str, Any]]
    weather_error: Optional[str]

    # SOP Policy Evaluations
    matched_sops: List[Dict[str, Any]]
    active_sop: Optional[Dict[str, Any]]
    sop_citation: Optional[str]

    # Guardrail & Retry State
    grounding_valid: Optional[bool]
    grounding_errors: List[str]
    retry_count: int

    # Output & Flow Control
    final_response: Optional[str]
    route: Optional[str]
    execution_trace: Annotated[List[str], operator.add]
