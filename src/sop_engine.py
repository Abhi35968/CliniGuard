from typing import Dict, Any, List, Optional
from src.config import SOPS_FILE_PATH
from src.policy.engine import (
    DeterministicPolicyEngine,
    SEVERITY_RANKS,
    ACTIVITY_SYNONYMS,
    DEMOGRAPHIC_SYNONYMS,
    INDOOR_ACTIVITIES,
)
from src.schemas.context import UserContext


class SOPEngine:
    """Decoupled Standard Operating Procedure (SOP) Rule & Evaluation Engine wrapper for backward compatibility."""

    def __init__(self, sops_path: str = SOPS_FILE_PATH):
        self.sops_path = sops_path
        self.engine = DeterministicPolicyEngine(sops_path=sops_path)

    def load_sops(self, force_reload: bool = False) -> List[Dict[str, Any]]:
        return self.engine.loader.load_policies(force_reload=force_reload)

    def evaluate(
        self,
        weather: Dict[str, Any],
        activity: Optional[str] = None,
        demographics: Optional[List[str]] = None,
        timeframe: Optional[str] = None,
        raw_query: str = "",
    ) -> List[Dict[str, Any]]:
        """Evaluates SOPs and returns sorted list of matched SOPs."""
        ctx = UserContext(
            activity=activity,
            demographics=demographics or [],
            timeframe=timeframe,
            location=None,
        )
        _, matched = self.engine.evaluate_risk(
            weather=weather,
            context=ctx,
            raw_query=raw_query,
        )
        return matched
