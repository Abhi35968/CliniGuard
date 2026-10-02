import json
from typing import Dict, Any, List, Optional
from src.schemas.context import UserContext
from src.llm.factory import get_llm_for_task
from src.llm.prompts import INTENT_EXTRACTION_SYSTEM_PROMPT
from src.llm.abstraction import LLMProvider


class ContextExtractionAgent:
    """Agent responsible for structured entity and intent extraction using LLM schema output."""

    def __init__(self, provider: Optional[LLMProvider] = None):
        self.provider = provider or get_llm_for_task(task_type="intent")

    def extract_context(
        self,
        query: str,
        existing_session_context: Dict[str, Any],
    ) -> UserContext:
        """
        Extracts structured UserContext from user query and merges with session state.
        """
        session_summary = (
            f"Existing Session Context -> "
            f"Location: {existing_session_context.get('location')}, "
            f"Activity: {existing_session_context.get('activity')}, "
            f"Timeframe: {existing_session_context.get('timeframe')}, "
            f"Demographics: {existing_session_context.get('demographics', [])}"
        )
        user_message = f"{session_summary}\nNew User Message: \"{query}\""

        def fallback_factory() -> UserContext:
            # Deterministic fallback extraction when LLM is unavailable
            from src.llm import extract_entities_fallback
            raw = extract_entities_fallback(query, existing_session_context)
            return UserContext(
                intent=raw.get("intent", "weather_safety_advisory"),
                location=raw.get("location"),
                activity=raw.get("activity"),
                timeframe=raw.get("timeframe"),
                demographics=raw.get("demographics", []),
                outdoor=True,
            )

        try:
            context = self.provider.structured_output(
                schema=UserContext,
                system_prompt=INTENT_EXTRACTION_SYSTEM_PROMPT,
                user_message=user_message,
                fallback_factory=fallback_factory,
            )

            # Preserve previous session context if current query didn't explicitly override it
            if not context.location and existing_session_context.get("location"):
                context.location = existing_session_context.get("location")
            if not context.activity and existing_session_context.get("activity"):
                context.activity = existing_session_context.get("activity")
            if not context.timeframe and existing_session_context.get("timeframe"):
                context.timeframe = existing_session_context.get("timeframe")
            if existing_session_context.get("demographics"):
                merged_demos = list(set(context.demographics + existing_session_context.get("demographics", [])))
                context.demographics = merged_demos

            return context
        except Exception as e:
            print(f"[ContextExtractionAgent] Error: {e}. Using fallback factory.")
            return fallback_factory()
