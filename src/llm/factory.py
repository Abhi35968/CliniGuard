from typing import Optional
from src.config import (
    LLM_PROVIDER,
    MODEL_NAME,
    INTENT_MODEL,
    GENERATION_MODEL,
    GUARDRAIL_MODEL,
    LLM_TEMPERATURE,
    GOOGLE_API_KEY,
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    GROQ_API_KEY,
    OPENROUTER_API_KEY,
)
from src.llm.abstraction import LLMProvider


def get_llm_for_task(task_type: str = "generation") -> LLMProvider:
    """
    Factory to instantiate configured LLMProvider based on task type and available API keys.
    Task types: 'intent', 'generation', 'guardrail', 'planner'.
    """
    provider = LLM_PROVIDER.lower()
    
    # Select task-specific model name
    if task_type == "intent":
        target_model = INTENT_MODEL
    elif task_type == "guardrail":
        target_model = GUARDRAIL_MODEL
    else:
        target_model = GENERATION_MODEL

    chat_model = None

    try:
        # 1. OpenRouter
        if (provider == "openrouter" or (not GOOGLE_API_KEY and not OPENAI_API_KEY and not GROQ_API_KEY)) and OPENROUTER_API_KEY:
            from langchain_openai import ChatOpenAI
            model = target_model if "/" in target_model else "meta-llama/llama-3.3-70b-instruct"
            chat_model = ChatOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
                model=model,
                temperature=LLM_TEMPERATURE,
            )

        # 2. Groq
        elif provider == "groq" and GROQ_API_KEY:
            from langchain_groq import ChatGroq
            valid_groq_models = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.6-27b", "groq/compound-mini", "llama-3.3-70b-versatile"]
            model = target_model if target_model in valid_groq_models else "llama-3.3-70b-versatile"
            chat_model = ChatGroq(
                model=model,
                groq_api_key=GROQ_API_KEY,
                temperature=LLM_TEMPERATURE,
            )

        # 3. Gemini
        elif provider == "gemini" and GOOGLE_API_KEY:
            from langchain_google_genai import ChatGoogleGenerativeAI
            model = target_model if "gemini" in target_model else "gemini-2.0-flash"
            chat_model = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=GOOGLE_API_KEY,
                temperature=LLM_TEMPERATURE,
            )

        # 4. OpenAI
        elif provider == "openai" and OPENAI_API_KEY:
            from langchain_openai import ChatOpenAI
            model = target_model if "gpt" in target_model else "gpt-4o-mini"
            chat_model = ChatOpenAI(
                model=model,
                api_key=OPENAI_API_KEY,
                temperature=LLM_TEMPERATURE,
            )

        # 5. Anthropic
        elif provider == "anthropic" and ANTHROPIC_API_KEY:
            from langchain_anthropic import ChatAnthropic
            model = target_model if "claude" in target_model else "claude-3-5-haiku-20241022"
            chat_model = ChatAnthropic(
                model=model,
                api_key=ANTHROPIC_API_KEY,
                temperature=LLM_TEMPERATURE,
            )

        # Fallback to OpenRouter if configured
        if not chat_model and OPENROUTER_API_KEY:
            from langchain_openai import ChatOpenAI
            chat_model = ChatOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
                model="meta-llama/llama-3.3-70b-instruct",
                temperature=LLM_TEMPERATURE,
            )

    except Exception as e:
        print(f"[LLM Factory] Could not initialize provider '{provider}' for task '{task_type}': {e}.")

    return LLMProvider(chat_model=chat_model, model_name=target_model if chat_model else "fallback")
