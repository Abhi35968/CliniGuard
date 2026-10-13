import json
import os
from typing import List, Dict, Any, Optional
from src.config import SOPS_FILE_PATH
from src.policy.models import SOPDefinition


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
            print(f"[PolicyLoader] Error loading policies from {self.sops_path}: {e}")
            return self._raw_cache or []
