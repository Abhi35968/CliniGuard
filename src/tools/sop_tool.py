from typing import Dict, Any, List, Optional
from src.tools.base import BaseTool
from src.retrieval.retriever import SOPRetriever
from src.schemas.context import UserContext


class SOPRetrieverTool(BaseTool):
    """Tool wrapper for semantic SOP policy retrieval."""

    name = "sop_retriever_tool"
    description = "Retrieves clinical SOP policies matching user query and context using semantic vector similarity."

    def __init__(self, retriever: Optional[SOPRetriever] = None):
        self.retriever = retriever or SOPRetriever()

    def run(self, query: str, context: UserContext, top_k: int = 5) -> Dict[str, Any]:
        """Retrieves top-k relevant SOPs for context."""
        sops = self.retriever.retrieve(query=query, context=context, top_k=top_k)
        return {
            "success": True,
            "retrieved_sops": sops,
            "count": len(sops),
        }
