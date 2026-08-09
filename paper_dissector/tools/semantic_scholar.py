"""Semantic Scholar API wrapper for evidence retrieval."""

from __future__ import annotations

import logging
import os
from typing import Any

from paper_dissector.config import SEMANTIC_SCHOLAR_API_KEY, SEMANTIC_SCHOLAR_BASE
from paper_dissector.tools._http import ResilientJSONClient

log = logging.getLogger(__name__)

_FIELDS = "paperId,title,year,authors,abstract,citationCount,url,externalIds"
_HEADERS = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY} if SEMANTIC_SCHOLAR_API_KEY else {}

# The unauthenticated pool is shared across all users and 429s aggressively;
# an API key raises the ceiling but does not remove it.
_client = ResilientJSONClient(
    name="s2",
    base_url=SEMANTIC_SCHOLAR_BASE,
    headers=_HEADERS,
    min_interval=float(os.getenv("S2_MIN_INTERVAL", "0.4" if SEMANTIC_SCHOLAR_API_KEY else "1.2")),
    timeout=float(os.getenv("S2_TIMEOUT", "30")),
    max_attempts=int(os.getenv("S2_MAX_ATTEMPTS", "3")),
    failure_threshold=int(os.getenv("S2_FAILURE_THRESHOLD", "3")),
    cooldown=float(os.getenv("S2_COOLDOWN", "120")),
    negative_ttl=int(os.getenv("S2_NEGATIVE_TTL", "300")),
)


def _get(path: str, params: dict, default: Any) -> Any:
    """Cached, throttled GET that degrades gracefully instead of raising."""
    return _client.get(path, params, default)


def search_papers(query: str, limit: int = 10, year_range: str | None = None) -> list[dict]:
    """
    Search Semantic Scholar for papers matching a query.

    Args:
        query: search string
        limit: max results (1-100)
        year_range: e.g. "2020-2025" or "2022-" for open-ended

    Returns:
        List of paper dicts with fields: paperId, title, year, authors, abstract, etc.
    """
    params = {
        "query": query,
        "limit": min(limit, 100),
        "fields": _FIELDS,
    }
    if year_range:
        params["year"] = year_range

    payload = _get("/paper/search", params, default={})
    return payload.get("data", []) if isinstance(payload, dict) else []


def get_paper_details(paper_id: str) -> dict | None:
    """Get full details for a single paper by Semantic Scholar ID or DOI."""
    return _get(
        f"/paper/{paper_id}",
        {"fields": _FIELDS + ",references,citations,tldr"},
        default=None,
    )


def get_citations(paper_id: str, limit: int = 20) -> list[dict]:
    """Get papers that cite the given paper (forward citations)."""
    payload = _get(
        f"/paper/{paper_id}/citations",
        {"fields": "paperId,title,year,abstract,citationCount", "limit": limit},
        default={},
    )
    rows = payload.get("data", []) if isinstance(payload, dict) else []
    return [c["citingPaper"] for c in rows if c.get("citingPaper")]


def get_references(paper_id: str, limit: int = 50) -> list[dict]:
    """Get papers cited by the given paper (backward references)."""
    payload = _get(
        f"/paper/{paper_id}/references",
        {"fields": "paperId,title,year,abstract,citationCount", "limit": limit},
        default={},
    )
    rows = payload.get("data", []) if isinstance(payload, dict) else []
    return [r["citedPaper"] for r in rows if r.get("citedPaper")]


def search_sota_for_task(task: str, metric: str, before_year: int, limit: int = 5) -> list[dict]:
    """
    Search for SOTA results on a task+metric, filtered to papers published before a cutoff.
    Used by the staleness checker to find baselines the paper should have compared against.
    """
    query = f"state of the art {task} {metric}"
    year_range = f"-{before_year}"
    return search_papers(query, limit=limit, year_range=year_range)


def lookup_paper_by_title(title: str) -> dict | None:
    """
    Find the Semantic Scholar record whose title best matches ``title``.

    Used as a source for a paper's publication year when neither Docling
    metadata nor the text itself yields one.
    """
    title = (title or "").strip()
    if len(title) < 8:
        return None

    payload = _get(
        "/paper/search/match",
        {"query": title, "fields": "paperId,title,year,authors,externalIds"},
        default=None,
    )
    if isinstance(payload, dict) and payload.get("data"):
        return payload["data"][0]

    # /match is strict; fall back to ordinary search and take the top hit.
    results = search_papers(title, limit=1)
    return results[0] if results else None
