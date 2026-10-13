import json
import re
from typing import Type, TypeVar, Optional, Any, Dict, List
from pydantic import BaseModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

T = TypeVar("T", bound=BaseModel)


class LLMProvider:
    """Wrapper abstraction over ChatModel providing structured output and text generation."""

    def __init__(self, chat_model: Optional[BaseChatModel] = None, model_name: str = "fallback"):
        self.chat_model = chat_model
        self.model_name = model_name

    def structured_output(
        self,
        schema: Type[T],
        system_prompt: str,
        user_message: str,
        fallback_factory: Optional[Any] = None,
    ) -> T:
        """Returns a validated Pydantic model instance from LLM response."""
        if not self.chat_model:
            if fallback_factory:
                return fallback_factory()
            raise RuntimeError(f"No LLM provider available for structured output extraction.")

        try:
            # Try native LangChain structured output first
            structured_llm = self.chat_model.with_structured_output(schema)
            res = structured_llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message),
            ])
            if isinstance(res, schema):
                return res
            elif isinstance(res, dict):
                return schema.model_validate(res)
        except Exception as e:
            # Fall back to explicit JSON prompt instructions & Pydantic parsing
            json_prompt = (
                f"{system_prompt}\n\n"
                f"CRITICAL: You MUST respond strictly with a valid JSON object matching the schema for {schema.__name__}.\n"
                f"JSON Schema: {json.dumps(schema.model_json_schema())}\n"
                f"Do NOT include markdown fences, preambles, or explanations outside the JSON."
            )
            try:
                response = self.chat_model.invoke([
                    SystemMessage(content=json_prompt),
                    HumanMessage(content=user_message),
                ])
                raw_text = response.content.strip()
                if raw_text.startswith("```"):
                    raw_text = re.sub(r"^```json\s*", "", raw_text, flags=re.IGNORECASE)
                    raw_text = re.sub(r"```$", "", raw_text).strip()
                
                parsed_json = json.loads(raw_text)
                return schema.model_validate(parsed_json)
            except Exception as parse_err:
                print(f"[LLMProvider] Structured extraction failed: {parse_err}. Using fallback factory if available.")
                if fallback_factory:
                    return fallback_factory()
                raise parse_err

    def generate(self, system_prompt: str, user_message: str) -> str:
        """Generates plain text response."""
        if not self.chat_model:
            raise RuntimeError("No LLM provider available for text generation.")
        
        response = self.chat_model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ])
        return response.content.strip()
