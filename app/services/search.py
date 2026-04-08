from __future__ import annotations
from typing import List
from app.core.config import TAVILY_API_KEY


def tavily_search(query: str, max_results: int = 5) -> List[dict]:
    """
    Search the web using Tavily.
    Returns an empty list if TAVILY_API_KEY is not set.
    """
    if not TAVILY_API_KEY:
        return []

    try:
        from langchain_community.tools.tavily_search import TavilySearchResults

        tool = TavilySearchResults(max_results=max_results)
        results = tool.invoke({"query": query}) or []

        out: List[dict] = []
        for r in results:
            out.append(
                {
                    "title": r.get("title") or "",
                    "url": r.get("url") or "",
                    "snippet": r.get("content") or r.get("snippet") or "",
                    "published_at": r.get("published_date") or r.get("published_at"),
                    "source": r.get("source"),
                }
            )
        return out

    except Exception:
        return []
