import os

# Base project root (one level up from code/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
except ImportError:
    pass

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.45"))
INTENT_CONF_THRESHOLD = float(os.getenv("INTENT_CONF_THRESHOLD", "0.60"))
EMBEDDING_MODEL = "all-MiniLM-L6-v2"