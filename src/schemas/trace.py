from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class ToolInvocationTrace(BaseModel):
    tool_name: str
    input_params: Dict[str, Any]
    output_summary: str
    execution_time_ms: float
    success: bool
    error: Optional[str] = None


class GuardrailCheckResult(BaseModel):
    guardrail_type: str  # "input", "output_schema", "grounding", "numeric", "policy_consistency"
    passed: bool
    details: str
    violations: List[str] = Field(default_factory=list)


class ExecutionTrace(BaseModel):
    """Structured telemetry trace log for system observability."""

    trace_id: str
    user_query: str
    nodes_executed: List[str] = Field(default_factory=list)
    tools_called: List[ToolInvocationTrace] = Field(default_factory=list)
    models_used: List[Dict[str, str]] = Field(default_factory=list)
    retrieved_sops: List[str] = Field(default_factory=list)
    policy_decision: Dict[str, Any] = Field(default_factory=dict)
    guardrail_results: List[GuardrailCheckResult] = Field(default_factory=list)
    latency_ms: Dict[str, float] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
