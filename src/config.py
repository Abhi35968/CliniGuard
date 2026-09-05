import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Root workspace directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# SOPs file path
SOPS_FILE_PATH = os.getenv("SOPS_FILE_PATH", str(ROOT_DIR / "data" / "sops.json"))
if not os.path.isabs(SOPS_FILE_PATH):
    SOPS_FILE_PATH = str(ROOT_DIR / SOPS_FILE_PATH)

# SQLite Database path for checkpoints and conversation persistence
DATABASE_PATH = os.getenv("DATABASE_PATH", str(ROOT_DIR / "data" / "medi_state.db"))
if not os.path.isabs(DATABASE_PATH):
    DATABASE_PATH = str(ROOT_DIR / DATABASE_PATH)

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").rstrip(";")

# Weather Endpoints (Open-Meteo)
OPEN_METEO_GEOCODING_URL = os.getenv(
    "OPEN_METEO_GEOCODING_URL", "https://geocoding-api.open-meteo.com/v1/search"
)
OPEN_METEO_FORECAST_URL = os.getenv(
    "OPEN_METEO_FORECAST_URL", "https://api.open-meteo.com/v1/forecast"
)

# Request timeout in seconds
WEATHER_API_TIMEOUT = 10.0
