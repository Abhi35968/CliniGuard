import time
import uuid
from typing import Dict, Any, List, Optional
from src.schemas.trace import ExecutionTrace, ToolInvocationTrace, GuardrailCheckResult


class ExecutionTracer:
    """Observability tracer tracking node transitions, tool invocations, model calls, and latency."""

    def __init__(self, trace_id: Optional[str] = None, user_query: str = ""):
        self.trace_id = trace_id or f"trace-{uuid.uuid4().hex[:10]}"
        self.user_query = user_query
        self.nodes_executed: List[str] = []
        self.tools_called: List[ToolInvocationTrace] = []
        self.models_used: List[Dict[str, str]] = []
        self.retrieved_sops: List[str] = []
        self.policy_decision: Dict[str, Any] = {}
        self.guardrail_results: List[GuardrailCheckResult] = []
        self.latency_ms: Dict[str, float] = {}
        self.errors: List[str] = []
        self._step_start_times: Dict[str, float] = {}

    def log_node_start(self, node_name: str):
        if node_name not in self.nodes_executed:
            self.nodes_executed.append(node_name)
        self._step_start_times[node_name] = time.time()

    def log_node_end(self, node_name: str):
        if node_name in self._step_start_times:
            elapsed = (time.time() - self._step_start_times[node_name]) * 1000.0
            self.latency_ms[node_name] = round(elapsed, 2)

    def log_tool_call(
        self,
        tool_name: str,
        input_params: Dict[str, Any],
        output_summary: str,
        execution_time_ms: float,
        success: bool,
        error: Optional[str] = None,
    ):
        # Sanitize parameters (do not log sensitive credentials)
        clean_inputs = {k: v for k, v in input_params.items() if "key" not in k.lower() and "secret" not in k.lower()}
        self.tools_called.append(
            ToolInvocationTrace(
                tool_name=tool_name,
                input_params=clean_inputs,
                output_summary=output_summary,
                execution_time_ms=round(execution_time_ms, 2),
                success=success,
                error=error,
            )
        )

    def log_model_call(self, task_type: str, model_name: str):
        self.models_used.append({"task_type": task_type, "model_name": model_name})

    def log_retrieved_sops(self, sop_ids: List[str]):
        self.retrieved_sops.extend(sop_ids)

    def log_policy_decision(self, decision_summary: Dict[str, Any]):
        self.policy_decision = decision_summary

    def log_guardrail_result(self, result: GuardrailCheckResult):
        self.guardrail_results.append(result)

    def log_error(self, error_msg: str):
        self.errors.append(error_msg)

    def export_trace(self) -> ExecutionTrace:
        return ExecutionTrace(
            trace_id=self.trace_id,
            user_query=self.user_query,
            nodes_executed=self.nodes_executed,
            tools_called=self.tools_called,
            models_used=self.models_used,
            retrieved_sops=list(set(self.retrieved_sops)),
            policy_decision=self.policy_decision,
            guardrail_results=self.guardrail_results,
            latency_ms=self.latency_ms,
            errors=self.errors,
        )
