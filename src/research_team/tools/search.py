from __future__ import annotations

import os
from typing import Protocol

import httpx
from bs4 import BeautifulSoup

SO360_ENDPOINT = "https://www.so.com/s"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


class SearchProvider(Protocol):
    def search(self, query: str, max_results: int = 5) -> list[dict]:
        """Return items shaped as {title, url, snippet}."""
        ...


def parse_so360(html: str, max_results: int) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    results: list[dict] = []

    for item in soup.select("li.res-list"):
        link = item.select_one("h3.res-title a")
        if link is None:
            continue

        url = (link.get("data-mdurl") or link.get("href") or "").strip()
        title = link.get_text(" ", strip=True)
        if not title or not url:
            continue

        summary = item.select_one(".res-list-summary")
        snippet = summary.get_text(" ", strip=True) if summary else ""
        results.append({"title": title, "url": url, "snippet": snippet})

        if len(results) >= max_results:
            break

    return results


class So360Provider:
    def __init__(self, max_attempts: int = 2, timeout: float = 15.0):
        self.max_attempts = max_attempts
        self.timeout = timeout

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        last_error: Exception | None = None
        for _ in range(self.max_attempts):
            try:
                response = httpx.get(
                    SO360_ENDPOINT,
                    params={"q": query},
                    headers={"User-Agent": USER_AGENT},
                    timeout=self.timeout,
                    follow_redirects=True,
                )
                response.raise_for_status()
                return parse_so360(response.text, max_results)
            except Exception as exc:  # noqa: BLE001
                last_error = exc

        raise RuntimeError(f"搜索失败: {last_error}")


class DuckDuckGoProvider:
    def __init__(self, max_attempts: int = 2):
        self.max_attempts = max_attempts

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        from ddgs import DDGS

        last_error: Exception | None = None
        for _ in range(self.max_attempts):
            try:
                with DDGS() as ddgs:
                    raw = ddgs.text(query, max_results=max_results)
                return [
                    {
                        "title": item.get("title", ""),
                        "url": item.get("href", ""),
                        "snippet": item.get("body", ""),
                    }
                    for item in raw
                ]
            except Exception as exc:  # noqa: BLE001
                last_error = exc

        raise RuntimeError(f"搜索失败: {last_error}")


def get_search_provider(name: str | None = None) -> SearchProvider:
    name = (name or os.getenv("SEARCH_PROVIDER", "so360")).lower()
    if name == "so360":
        return So360Provider()
    if name == "duckduckgo":
        return DuckDuckGoProvider()
    raise ValueError(f"未知的搜索源: {name}")
