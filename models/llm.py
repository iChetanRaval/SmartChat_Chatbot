"""
models/llm.py
LangChain Groq wrapper — the only LLM provider used in this project.
"""

import logging
from typing import Optional

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)


def build_system_prompt(mode: str = "concise") -> str:
    """Return a system prompt based on the selected response mode."""
    if mode == "concise":
        return (
            "You are a helpful, concise assistant. "
            "Respond in 2-4 sentences maximum. Be direct and to the point. "
            "Omit pleasantries and lengthy explanations."
        )
    else:  # detailed
        return (
            "You are a thorough, knowledgeable assistant. "
            "Provide detailed, well-structured responses. "
            "Use bullet points, examples, and explain your reasoning step by step."
        )


def get_chatgroq_model(model_name: Optional[str] = None) -> BaseChatModel:
    """Return a LangChain ChatGroq model instance."""
    try:
        from langchain_groq import ChatGroq
        from config.config import GROQ_API_KEY, GROQ_MODEL
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file: GROQ_API_KEY=your_key_here"
            )
        return ChatGroq(
            api_key=GROQ_API_KEY,
            model=model_name or GROQ_MODEL,
            temperature=0.7,
        )
    except ImportError as exc:
        raise ImportError("Run: pip install langchain-groq") from exc
    except Exception as exc:
        logger.error("Groq model init failed: %s", exc)
        raise