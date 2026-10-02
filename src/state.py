from typing import Dict, Any, List, Optional, Annotated
from typing_extensions import TypedDict
import operator

from src.schemas.context import UserContext
from src.schemas.risk import RiskAssessment


class AgentState(TypedDict):
    """LangGraph state representation for CliniGuard Weather-Advisory Assistant."""

    # Session & Chat history
    session_id: str
    messages: Annotated[List[Dict[str, Any]], operator.add]
    raw_query: str

    # Structured User Context (Pydantic object & flattened fields for backward compatibility)
    user_context: Optional[Dict[str, Any]]
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

    # SOP Policy & Risk Engine Evaluations
    retrieved_sops: List[Dict[str, Any]]
    matched_sops: List[Dict[str, Any]]
    active_sop: Optional[Dict[str, Any]]
    risk_assessment: Optional[Dict[str, Any]]
    sop_citation: Optional[str]

    # Guardrail & Retry State
    input_guardrail_passed: Optional[bool]
    grounding_valid: Optional[bool]
    grounding_errors: List[str]
    guardrail_results: List[Dict[str, Any]]
    retry_count: int

    # Output & Flow Control
    final_response: Optional[str]
    route: Optional[str]
    execution_trace: Annotated[List[str], operator.add]
