"""Optional Tavily-backed financial news search with a no-key fallback."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None


class FinancialNewsSearch:
    """Fetch and cache recent financial news when Tavily is configured."""

    def __init__(self) -> None:
        self._client = None
        self._cache: dict[str, tuple[datetime, list[dict[str, Any]]]] = {}
        api_key = os.getenv("TAVILY_API_KEY")
        if api_key and TavilyClient is not None:
            self._client = TavilyClient(api_key=api_key)

    def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        if self._client is None:
            return []

        cache_key = f"{query}:{max_results}"
        cached = self._cache.get(cache_key)
        if cached and datetime.now() - cached[0] < timedelta(hours=1):
            return cached[1]

        try:
            response = self._client.search(
                query=f"{query} financial market news",
                search_depth="advanced",
                max_results=max_results,
                days=7,
            )
            results = [
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "published_date": item.get("published_date", ""),
                    "relevance_score": item.get("score", 0),
                }
                for item in response.get("results", [])
            ]
            self._cache[cache_key] = (datetime.now(), results)
            return results
        except Exception:
            return []


_news_search = FinancialNewsSearch()


def search_financial_news(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Return recent news results, or an empty list when search is unavailable."""
    return _news_search.search(query, max_results=max_results)
