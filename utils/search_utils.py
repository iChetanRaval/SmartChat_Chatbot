"""
utils/search_utils.py
Real-time web search integration.
Supports: DuckDuckGo (free, no key), Serper.dev, Tavily.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ── Result dataclass ───────────────────────────────────────────────────────────

class SearchResult:
    def __init__(self, title: str, url: str, snippet: str):
        self.title = title
        self.url = url
        self.snippet = snippet

    def __repr__(self) -> str:
        return f"SearchResult(title={self.title!r})"


# ── DuckDuckGo (free, no API key) ──────────────────────────────────────────────

def _search_duckduckgo(query: str, max_results: int = 5) -> list[SearchResult]:
    try:
        from duckduckgo_search import DDGS  # type: ignore
        results: list[SearchResult] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(SearchResult(
                    title=r.get("title", ""),
                    url=r.get("href", ""),
                    snippet=r.get("body", ""),
                ))
        return results
    except ImportError as exc:
        raise ImportError(
            "duckduckgo-search is required. Install: pip install duckduckgo-search"
        ) from exc
    except Exception as exc:
        logger.error("DuckDuckGo search failed: %s", exc)
        raise RuntimeError(f"DuckDuckGo search error: {exc}") from exc


# ── Serper.dev ─────────────────────────────────────────────────────────────────

def _search_serper(query: str, api_key: str, max_results: int = 5) -> list[SearchResult]:
    try:
        import requests  # type: ignore
        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
        payload = {"q": query, "num": max_results}
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        results: list[SearchResult] = []
        for item in data.get("organic", [])[:max_results]:
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
            ))
        return results
    except Exception as exc:
        logger.error("Serper search failed: %s", exc)
        raise RuntimeError(f"Serper search error: {exc}") from exc


# ── Tavily ─────────────────────────────────────────────────────────────────────

def _search_tavily(query: str, api_key: str, max_results: int = 5) -> list[SearchResult]:
    try:
        from tavily import TavilyClient  # type: ignore
        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, max_results=max_results)
        results: list[SearchResult] = []
        for item in response.get("results", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
            ))
        return results
    except ImportError as exc:
        raise ImportError("tavily-python is required. Install: pip install tavily-python") from exc
    except Exception as exc:
        logger.error("Tavily search failed: %s", exc)
        raise RuntimeError(f"Tavily search error: {exc}") from exc


# ── Public API ─────────────────────────────────────────────────────────────────

def web_search(
    query: str,
    provider: Optional[str] = None,
    max_results: int = 5,
) -> list[SearchResult]:
    """
    Perform a real-time web search.

    Args:
        query:       The search query.
        provider:    "duckduckgo" | "serper" | "tavily" (falls back to config default).
        max_results: Maximum number of results to return.

    Returns:
        List of SearchResult objects.
    """
    from config.config import SEARCH_PROVIDER, SERPER_API_KEY, TAVILY_API_KEY

    provider = (provider or SEARCH_PROVIDER).lower()

    if provider == "duckduckgo":
        return _search_duckduckgo(query, max_results)

    elif provider == "serper":
        if not SERPER_API_KEY:
            raise ValueError("SERPER_API_KEY is not configured.")
        return _search_serper(query, SERPER_API_KEY, max_results)

    elif provider == "tavily":
        if not TAVILY_API_KEY:
            raise ValueError("TAVILY_API_KEY is not configured.")
        return _search_tavily(query, TAVILY_API_KEY, max_results)

    else:
        raise ValueError(f"Unknown search provider: '{provider}'. Use duckduckgo / serper / tavily.")


def format_search_results(results: list[SearchResult]) -> str:
    """Format search results into a readable context block for the LLM."""
    if not results:
        return "No search results found."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. **{r.title}**\n   {r.snippet}\n   Source: {r.url}")
    return "\n\n".join(lines)