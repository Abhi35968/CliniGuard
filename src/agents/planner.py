from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from src.schemas.context import UserContext
from src.llm.factory import get_llm_for_task
from src.llm.prompts import PLANNER_SYSTEM_PROMPT
from src.llm.abstraction import LLMProvider


class ExecutionPlan(BaseModel):
    """Execution plan formulated by PlannerAgent."""

    needs_clarification: bool = Field(
        default=False,
        description="True if mandatory parameter (e.g. location) is missing.",
    )
    clarification_target: Optional[str] = Field(
        default=None,
        description="Parameter needing clarification, e.g. 'location'.",
    )
    tools_to_run: List[str] = Field(
        default_factory=list,
        description="List of tool names to run e.g. ['weather_tool', 'sop_retriever_tool'].",
    )
    reasoning: str = Field(
        default="",
        description="Reasoning behind tool planning.",
    )


class PlannerAgent:
    """Planner Agent determining tool execution flow based on UserContext."""

    def __init__(self, provider: Optional[LLMProvider] = None):
        self.provider = provider or get_llm_for_task(task_type="planner")

    def plan(self, context: UserContext, query: str) -> ExecutionPlan:
        """Formulates ExecutionPlan for request."""
        # 1. Deterministic safety rule: Outdoor weather advisory MUST have a location
        if not context.location:
            return ExecutionPlan(
                needs_clarification=True,
                clarification_target="location",
                tools_to_run=[],
                reasoning="Missing target location required for fetching live meteorological telemetry.",
            )

        # 2. Indoor non-weather query check
        if context.outdoor is False:
            return ExecutionPlan(
                needs_clarification=False,
                clarification_target=None,
                tools_to_run=["weather_tool"],
                reasoning="Indoor activity request. Fetch telemetry for general ambient notice.",
            )

        # 3. Standard Plan
        return ExecutionPlan(
            needs_clarification=False,
            clarification_target=None,
            tools_to_run=["weather_tool", "sop_retriever_tool"],
            reasoning=f"Location '{context.location}' available. Execute weather telemetry fetch and semantic SOP retrieval.",
        )
