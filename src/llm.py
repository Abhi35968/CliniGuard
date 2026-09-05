import os
import json
import re
from typing import Optional, Dict, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from src.config import (
    LLM_PROVIDER,
    MODEL_NAME,
    LLM_TEMPERATURE,
    GOOGLE_API_KEY,
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    GROQ_API_KEY,
    OPENROUTER_API_KEY,
)


def get_llm() -> Optional[BaseChatModel]:
    """Factory to instantiate configured ChatModel based on provider and available keys."""
    provider = LLM_PROVIDER.lower()

    try:
        # OpenRouter
        if (provider == "openrouter" or not GOOGLE_API_KEY and not OPENAI_API_KEY and not GROQ_API_KEY) and OPENROUTER_API_KEY:
            from langchain_openai import ChatOpenAI
            model = MODEL_NAME if "/" in MODEL_NAME else "meta-llama/llama-3.3-70b-instruct"
            return ChatOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
                model=model,
                temperature=LLM_TEMPERATURE,
            )

        # Groq
        elif provider == "groq" and GROQ_API_KEY:
            from langchain_groq import ChatGroq
            valid_groq_models = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.6-27b", "groq/compound-mini"]
            model = MODEL_NAME if MODEL_NAME in valid_groq_models else "openai/gpt-oss-120b"
            return ChatGroq(
                model=model,
                groq_api_key=GROQ_API_KEY,
                temperature=LLM_TEMPERATURE,
            )

        # Gemini
        elif provider == "gemini" and GOOGLE_API_KEY:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=MODEL_NAME if "gemini" in MODEL_NAME else "gemini-2.0-flash",
                google_api_key=GOOGLE_API_KEY,
                temperature=LLM_TEMPERATURE,
            )

        # OpenAI
        elif provider == "openai" and OPENAI_API_KEY:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=MODEL_NAME if "gpt" in MODEL_NAME else "gpt-4o-mini",
                api_key=OPENAI_API_KEY,
                temperature=LLM_TEMPERATURE,
            )

        # Anthropic
        elif provider == "anthropic" and ANTHROPIC_API_KEY:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=MODEL_NAME if "claude" in MODEL_NAME else "claude-3-5-haiku-20241022",
                api_key=ANTHROPIC_API_KEY,
                temperature=LLM_TEMPERATURE,
            )

        # Fallback to OpenRouter if available
        if OPENROUTER_API_KEY:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
                model="meta-llama/llama-3.3-70b-instruct",
                temperature=LLM_TEMPERATURE,
            )

    except Exception as e:
        print(f"[LLM Factory] Could not initialize LLM provider '{provider}': {e}. Using deterministic engine.")

    return None


def extract_entities_fallback(query: str, existing_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    High-fidelity deterministic entity extractor when LLM is unavailable or for regex verification.
    """
    q = query.lower()
    
    # 1. Location extraction
    location = None
    # Common Indian & Global test cities
    cities = [
        "bhopal", "new delhi", "delhi", "mumbai", "bengaluru", "bangalore", "chennai",
        "kolkata", "hyderabad", "pune", "ahmedabad", "jaipur", "lucknow", "chandigarh",
        "goa", "shimla", "manali", "surat", "patna", "indore", "london", "new york", "berlin", "tokyo"
    ]
    # Check known cities first
    for c in cities:
        # Match whole word
        if re.search(r"\b" + re.escape(c) + r"\b", q):
            location = c.title()
            break

    # If no known city, extract spatial preposition candidate (in/at/near/around)
    if not location:
        loc_match = re.search(r"\b(?:in|at|around|near)\s+([A-Za-z0-9_\-]+(?:\s+[A-Za-z0-9_\-]+)?)", query, re.IGNORECASE)
        if loc_match:
            cand = loc_match.group(1).strip()
            # Exclude stopwords, articles, or timeframe words
            stopwords = {"the", "a", "an", "my", "our", "this", "that", "today", "tomorrow", "morning", "evening", "afternoon", "night"}
            words = cand.lower().split()
            cleaned_words = [w for w in words if w not in stopwords]
            if cleaned_words:
                location = " ".join(cleaned_words).title()

    # If still no location in current query, retain from session context
    if not location:
        location = existing_context.get("location")

    # 2. Activity extraction
    activity = existing_context.get("activity")
    activity_map = {
        "cycling": ["cycling", "cycle", "bike", "biking", "bicycle", "pedal", "ride"],
        "running": ["running", "run", "jogging", "jog", "marathon", "sprint"],
        "walking": ["walking", "walk", "stroll", "pedestrian"],
        "picnic": ["picnic", "park outing", "barbecue", "outdoor lunch", "garden party"],
        "playground": ["playground", "swings", "slides", "play area", "park"],
        "two-wheeler": ["two-wheeler", "scooter", "motorcycle", "bike commute", "scooty"],
        "travel": ["travel", "highway", "drive", "road trip", "long drive", "commute"],
        "dog walking": ["dog", "puppy", "pet", "canine"],
        "hiking": ["hiking", "hike", "trekking", "trek", "camping"],
        "swimming": ["swimming", "swim", "pool"],
        "chess": ["chess", "board game", "cards"],
        "indoor": ["indoor", "living room", "painting", "reading"],
    }
    for act_key, keywords in activity_map.items():
        if any(kw in q for kw in keywords):
            activity = act_key
            break

    # 3. Timeframe extraction
    timeframe = existing_context.get("timeframe") or "today"
    time_keywords = ["this evening", "evening", "tomorrow morning", "morning", "afternoon", "midday", "tonight", "night", "tomorrow", "today", "now"]
    for tk in time_keywords:
        if tk in q:
            timeframe = tk
            break

    # 4. Demographics
    demographics = list(existing_context.get("demographics", []))
    demo_map = {
        "children": ["kid", "kids", "child", "children", "toddler", "toddlers", "baby", "babies", "daughter", "son"],
        "elderly": ["elderly", "senior", "seniors", "grandparents", "grandfather", "grandmother", "cardiac"],
        "pets": ["dog", "dogs", "puppy", "pup", "pet", "pets", "cat"],
    }
    for demo_key, keywords in demo_map.items():
        if any(kw in q for kw in keywords):
            if demo_key not in demographics:
                demographics.append(demo_key)

    return {
        "location": location,
        "activity": activity,
        "timeframe": timeframe,
        "demographics": demographics,
        "intent": "safety_advisory",
    }
