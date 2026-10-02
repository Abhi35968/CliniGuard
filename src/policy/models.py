from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

SEVERITY_RANKS: Dict[str, int] = {
    "CRITICAL": 4,
    "HIGH": 3,
    "MODERATE": 2,
    "LOW": 1,
    "INFORMATIONAL": 0,
}


class TimeWindow(BaseModel):
    start_hour: int = 0
    end_hour: int = 24


class SOPConditionRule(BaseModel):
    field: str
    op: str  # ">=", "<=", ">", "<", "==", "!=", "in"
    value: Any


class SOPConditionTree(BaseModel):
    type: str = "numeric"  # "numeric", "composite", "fuzzy_comfort"
    operator: str = "AND"  # "AND", "OR"
    rules: List[Any] = Field(default_factory=list)
    evaluation: Optional[Dict[str, Any]] = None


class SOPDefinition(BaseModel):
    id: str
    title: str
    category: str
    target_activities: List[str]
    target_demographics: List[str]
    severity: str  # "CRITICAL", "HIGH", "MODERATE", "LOW"
    priority: int = 0
    time_window: Optional[TimeWindow] = None
    conditions: Dict[str, Any]
    advisory_lead: Optional[str] = None
    guidance: str
    precautions: List[str] = Field(default_factory=list)
