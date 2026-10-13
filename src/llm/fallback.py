import re
from typing import Dict, Any


def extract_entities_fallback(query: str, existing_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    High-fidelity deterministic entity extractor when LLM is unavailable or for fallback verification.
    """
    q = query.lower()
    
    # 1. Location extraction
    location = None
    cities = [
        "bhopal", "new delhi", "delhi", "mumbai", "bengaluru", "bangalore", "chennai",
        "kolkata", "hyderabad", "pune", "ahmedabad", "jaipur", "lucknow", "chandigarh",
        "goa", "shimla", "manali", "surat", "patna", "indore", "london", "new york", "berlin", "tokyo"
    ]
    for c in cities:
        if re.search(r"\b" + re.escape(c) + r"\b", q):
            location = c.title()
            break

    if not location:
        loc_match = re.search(r"\b(?:in|at|around|near)\s+([A-Za-z0-9_\-]+(?:\s+[A-Za-z0-9_\-]+)?)", query, re.IGNORECASE)
        if loc_match:
            cand = loc_match.group(1).strip()
            stopwords = {"the", "a", "an", "my", "our", "this", "that", "today", "tomorrow", "morning", "evening", "afternoon", "night"}
            words = cand.lower().split()
            cleaned_words = [w for w in words if w not in stopwords]
            if cleaned_words:
                location = " ".join(cleaned_words).title()

    if not location:
        location = existing_context.get("location")

    # 2. Activity extraction
    activity = existing_context.get("activity")
    activity_map = {
        "dog walking": ["dog walking", "dog walk", "pet walk", "walk my dog", "puppy", "canine", "retriever", "golden retriever", "taking pet out", "dog"],
        "cycling": ["cycling", "cycle", "bike", "biking", "bicycle", "pedal", "ride"],
        "running": ["running", "run", "jogging", "jog", "marathon", "sprint"],
        "walking": ["walking", "walk", "stroll", "pedestrian"],
        "picnic": ["picnic", "park outing", "barbecue", "outdoor lunch", "garden party"],
        "playground": ["playground", "swings", "slides", "play area", "park"],
        "two-wheeler": ["two-wheeler", "scooter", "motorcycle", "bike commute", "scooty"],
        "travel": ["travel", "highway", "drive", "road trip", "long drive", "commute"],
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
        "pets": ["dog", "dogs", "puppy", "pup", "pet", "pets", "cat", "retriever", "canine", "animal"],
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
        "intent": "weather_safety_advisory",
    }
