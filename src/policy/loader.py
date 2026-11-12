import json
import os
from typing import List, Dict, Any, Optional
from src.config import SOPS_FILE_PATH
from src.policy.models import SOPDefinition
from src.observability import get_logger

logger = get_logger("policy_loader")



class PolicyLoader:
    """Loader and validator for Standard Operating Procedure (SOP) policy configurations."""

    def __init__(self, sops_path: str = SOPS_FILE_PATH):
        self.sops_path = sops_path
        self._cache: Optional[List[SOPDefinition]] = None
        self._raw_cache: Optional[List[Dict[str, Any]]] = None
        self._last_mtime: float = 0.0

    def load_policies(self, force_reload: bool = False) -> List[Dict[str, Any]]:
        """Loads raw dictionary SOP definitions from JSON file. Auto-reloads if modified."""
        if not os.path.exists(self.sops_path):
            return []

        try:
            mtime = os.path.getmtime(self.sops_path)
            if force_reload or self._raw_cache is None or mtime > self._last_mtime:
                with open(self.sops_path, "r", encoding="utf-8") as f:
                    self._raw_cache = json.load(f)
                self._last_mtime = mtime
            return self._raw_cache or []
        except Exception as e:
            logger.error("Error loading policies from %s: %s", self.sops_path, e)
            return self._raw_cache or []

    def load_override_policies(self, force_reload: bool = False) -> List[Dict[str, Any]]:
        """Loads only severe-weather override policies (category='severe_override' or wildcard target_activities)."""
        policies = self.load_policies(force_reload=force_reload)
        return [
            s for s in policies
            if s.get("category") == "severe_override" or "*" in s.get("target_activities", [])
        ]
