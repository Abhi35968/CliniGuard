from src.llm.abstraction import LLMProvider
from src.llm.factory import get_llm_for_task
from src.llm.fallback import extract_entities_fallback


def get_llm():
    """Backward compatibility wrapper returning BaseChatModel."""
    provider = get_llm_for_task("generation")
    return provider.chat_model


__all__ = [
    "LLMProvider",
    "get_llm_for_task",
    "get_llm",
    "extract_entities_fallback",
]
