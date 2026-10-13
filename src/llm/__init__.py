from src.llm.abstraction import LLMProvider
from src.llm.factory import get_llm_for_task
from src.llm.fallback import extract_entities_fallback
from src.llm.prompts import (
    INTENT_EXTRACTION_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    ADVISORY_GENERATION_SYSTEM_PROMPT,
    INPUT_GUARDRAIL_SYSTEM_PROMPT,
    OUTPUT_GROUNDING_SYSTEM_PROMPT,
)


def get_llm():
    """Backward compatibility wrapper returning BaseChatModel."""
    provider = get_llm_for_task("generation")
    return provider.chat_model


__all__ = [
    "LLMProvider",
    "get_llm_for_task",
    "get_llm",
    "extract_entities_fallback",
    "INTENT_EXTRACTION_SYSTEM_PROMPT",
    "PLANNER_SYSTEM_PROMPT",
    "ADVISORY_GENERATION_SYSTEM_PROMPT",
    "INPUT_GUARDRAIL_SYSTEM_PROMPT",
    "OUTPUT_GROUNDING_SYSTEM_PROMPT",
]
