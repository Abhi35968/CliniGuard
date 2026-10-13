from typing import List, Optional
from pydantic import BaseModel, Field


class UserContext(BaseModel):
    """Structured extraction schema of user query context."""

    intent: str = Field(
        default="weather_safety_advisory",
        description="Detected user intent, e.g., 'weather_safety_advisory', 'location_clarification', 'general_query'.",
    )
    location: Optional[str] = Field(
        default=None,
        description="Target city or location name extracted from query or session memory.",
    )
    activity: Optional[str] = Field(
        default=None,
        description="Specific outdoor or indoor activity, e.g., 'cycling', 'running', 'walking', 'picnic', 'playground', 'chess'.",
    )
    timeframe: Optional[str] = Field(
        default=None,
        description="Requested timeframe e.g., 'today', 'this evening', 'tomorrow morning', 'afternoon', 'now'.",
    )
    demographics: List[str] = Field(
        default_factory=list,
        description="Vulnerable demographic groups mentioned, e.g., ['children', 'elderly', 'pets', 'cardiac'].",
    )
    outdoor: Optional[bool] = Field(
        default=True,
        description="Whether the activity is outdoors (True) or indoors (False).",
    )
    constraints: List[str] = Field(
        default_factory=list,
        description="Specific user constraints or preferences mentioned in the query.",
    )
