"""
utils/chat_utils.py
Prompt assembly: stitches together RAG context, search results, and the user message.
"""

from typing import Optional


def build_augmented_prompt(
    user_message: str,
    rag_context: Optional[str] = None,
    search_context: Optional[str] = None,
) -> str:
    """
    Construct the final prompt sent to the LLM, optionally prepending
    RAG context and/or live web search results.

    Args:
        user_message:   Raw question from the user.
        rag_context:    Pre-formatted string from retrieved document chunks.
        search_context: Pre-formatted string from web search results.

    Returns:
        A single string to pass as the `prompt` argument to get_llm_response().
    """
    sections: list[str] = []

    if rag_context:
        sections.append(
            "### Relevant Knowledge Base Excerpts\n"
            "Use the excerpts below to help answer the question. "
            "Cite the source label when referencing them.\n\n"
            + rag_context
        )

    if search_context:
        sections.append(
            "### Live Web Search Results\n"
            "The following results were retrieved in real time. "
            "Prioritise them for recent or factual information.\n\n"
            + search_context
        )

    sections.append(f"### User Question\n{user_message}")
    return "\n\n".join(sections)


def trim_history(history: list[dict], max_turns: int = 10) -> list[dict]:
    """Keep only the most recent N full turns (user + assistant pairs)."""
    return history[-(max_turns * 2):]