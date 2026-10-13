from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class RiskAssessment(BaseModel):
    """Structured intermediate risk evaluation object produced deterministically by Policy Engine."""

    risk_level: str = Field(
        default="INFORMATIONAL",
        description="Overall severity level: 'CRITICAL', 'HIGH', 'MODERATE', 'LOW', 'INFORMATIONAL', 'UNKNOWN'.",
    )
    triggered_policies: List[str] = Field(
        default_factory=list,
        description="List of matched SOP identifiers e.g., ['SOP-EXER-001', 'SOP-OVERRIDE-001'].",
    )
    primary_sop_id: Optional[str] = Field(
        default=None,
        description="Primary governing SOP ID.",
    )
    primary_sop_title: Optional[str] = Field(
        default=None,
        description="Title of primary governing SOP.",
    )
    reasons: List[str] = Field(
        default_factory=list,
        description="Detailed technical/meteorological reasons for triggering these policies.",
    )
    mandatory_actions: List[str] = Field(
        default_factory=list,
        description="Mandatory safety precautions required by active SOPs.",
    )
    prohibited_actions: List[str] = Field(
        default_factory=list,
        description="Explicitly prohibited activities or behaviors under active SOPs.",
    )
    telemetry_used: Dict[str, Any] = Field(
        default_factory=dict,
        description="Verified meteorological telemetry values evaluated.",
    )
    sop_citation: str = Field(
        default="NO_SOP_APPLICABLE",
        description="Formatted citation string for provenance.",
    )
