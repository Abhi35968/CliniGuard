from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseTool(ABC):
    """Abstract Base Class for all CliniGuard Tools."""

    name: str
    description: str

    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        """Executes the tool with given arguments and returns a structured output payload."""
        pass
