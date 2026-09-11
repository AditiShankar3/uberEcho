"""
config.py
Configuration parameters and threshold settings.
Reads from .env if present, with graceful fallbacks.
"""

import os

# Try loading from .env if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# Thresholds
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.45"))
INTENT_CONF_THRESHOLD = float(os.getenv("INTENT_CONF_THRESHOLD", "0.60"))
EMBEDDING_MODEL = "all-MiniLM-L6-v2"