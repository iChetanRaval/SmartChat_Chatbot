"""
config/config.py
All settings loaded from environment variables / .env file.
Never hardcode API keys here.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env file automatically

# ── Groq ───────────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# ── Web Search ────────────────────────────────────────────────────────────────
# "duckduckgo" works with no API key — recommended default
SEARCH_PROVIDER: str = os.getenv("SEARCH_PROVIDER", "duckduckgo")
SERPER_API_KEY: str = os.getenv("SERPER_API_KEY", "")
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

# ── RAG / Embeddings ──────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "3"))

# ── App ───────────────────────────────────────────────────────────────────────
APP_TITLE: str = "SmartChat"
DEFAULT_RESPONSE_MODE: str = "concise"
MAX_HISTORY: int = 20