"""OpenAlex API wrapper for evidence retrieval.

OpenAlex is the keyless alternative to Semantic Scholar. It exposes the same
things the evidence hunter needs — title, year, DOI, citation count, abstract —
so results are normalised into the Semantic Scholar record shape and the rest of
the pipeline does not care which backend produced them.
"""

from __future__ import annotations

import logging
import os

from paper_dissector.config import OPENALEX_API_KEY, OPENALEX_BASE, OPENALEX_MAILTO
from paper_dissector.tools._http import ResilientJSONClient

log = logging.getLogger(__name__)

_SELECT = (
    "id,doi,title,display_name,publication_year,cited_by_count,"
    "abstract_inverted_index,authorships,primary_location"
)

# A contact address earns the faster "polite pool"; without one we are on the
# shared common pool, which still works but is slower under load.
_headers = {"User-Agent": "paper-dissector"}
if OPENALEX_MAILTO:
    _headers["User-Agent"] = f"paper-dissector (mailto:{OPENALEX_MAILTO})"

_client = ResilientJSONClient(
    name="openalex",
    base_url=OPENALEX_BASE,
    headers=_headers,
    # OpenAlex throttles bursts, so space requests out. Give up quickly though:
    # when it pauses anonymous search there is a working fallback behind us in
    # the chain, and grinding through retries just delays reaching it.
    min_interval=float(os.getenv("OPENALEX_MIN_INTERVAL", "1.1")),
    timeout=float(os.getenv("OPENALEX_TIMEOUT", "30")),
    max_attempts=int(os.getenv("OPENALEX_MAX_ATTEMPTS", "2")),
    failure_threshold=int(os.getenv("OPENALEX_FAILURE_THRESHOLD", "2")),
    cooldown=float(os.getenv("OPENALEX_COOLDOWN", "90")),
)


def _base_params() -> dict:
    params = {"select": _SELECT}
    if OPENALEX_MAILTO:
        params["mailto"] = OPENALEX_MAILTO
    if OPENALEX_API_KEY:
        # A key exempts us from the anonymous-search pause OpenAlex applies
        # when its search cluster is under load.
        params["api_key"] = OPENALEX_API_KEY
    return params


def _reconstruct_abstract(inverted: dict | None) -> str:
    """
    Rebuild plain text from OpenAlex's inverted index.

    The index maps each word to the positions it occupies, so we place words
    back at their positions and join.
    """
    if not isinstance(inverted, dict) or not inverted:
        return ""

    positioned: list[tuple[int, str]] = []
    for word, positions in inverted.items():
        if not isinstance(positions, list):
            continue
        for pos in positions:
            if isinstance(pos, int):
                positioned.append((pos, word))

    if not positioned:
        return ""
    positioned.sort(key=lambda x: x[0])
    return " ".join(word for _, word in positioned)


def _short_id(openalex_id: str) -> str:
    """'https://openalex.org/W123' -> 'W123'."""
    return (openalex_id or "").rstrip("/").rsplit("/", 1)[-1]


def _normalise(work: dict) -> dict:
    """Convert an OpenAlex work into the Semantic Scholar record shape."""
    doi_url = work.get("doi") or ""
    doi = doi_url.replace("https://doi.org/", "") if doi_url else None

    authors = []
    for authorship in (work.get("authorships") or [])[:10]:
        author = (authorship or {}).get("author") or {}
        name = author.get("display_name")
        if name:
            authors.append({"name": name})

    landing = ((work.get("primary_location") or {}).get("landing_page_url")) or None

    return {
        "paperId": _short_id(work.get("id", "")),
        "title": work.get("title") or work.get("display_name") or "",
        "year": work.get("publication_year"),
        "authors": authors,
        "abstract": _reconstruct_abstract(work.get("abstract_inverted_index")),
        "citationCount": work.get("cited_by_count"),
        "url": landing or (doi_url or None),
        "externalIds": {"DOI": doi} if doi else {},
    }


def search_papers(query: str, limit: int = 10, year_range: str | None = None) -> list[dict]:
    """
    Search OpenAlex for works matching a query.

    Args:
        query: search string
        limit: max results (1-200)
        year_range: "2020-2025", "2022-" or "-2015", matching the
            Semantic Scholar convention used elsewhere in the codebase.

    Returns:
        List of paper dicts in the Semantic Scholar record shape.
    """
    query = (query or "").strip()
    if not query:
        return []

    params = _base_params()
    params["search"] = query
    params["per-page"] = min(max(limit, 1), 200)

    filters = _year_filter(year_range)
    if filters:
        params["filter"] = filters

    payload = _client.get("/works", params, default={})
    results = payload.get("results", []) if isinstance(payload, dict) else []
    return [_normalise(w) for w in results if isinstance(w, dict)]


def _year_filter(year_range: str | None) -> str | None:
    """Translate a 'YYYY-YYYY' / 'YYYY-' / '-YYYY' range into an OpenAlex filter."""
    if not year_range:
        return None
    value = year_range.strip()

    try:
        if value.startswith("-"):
            return f"publication_year:<{int(value[1:])}"
        if value.endswith("-"):
            return f"publication_year:>{int(value[:-1]) - 1}"
        if "-" in value:
            start, end = value.split("-", 1)
            return f"publication_year:{int(start)}-{int(end)}"
        return f"publication_year:{int(value)}"
    except ValueError:
        log.debug("could not parse year_range %r; ignoring", year_range)
        return None


def get_paper_details(paper_id: str) -> dict | None:
    """Get full details for a single work by OpenAlex ID or DOI."""
    if not paper_id:
        return None
    ident = paper_id if paper_id.upper().startswith("W") else f"doi:{paper_id}"
    payload = _client.get(f"/works/{ident}", _base_params(), default=None)
    return _normalise(payload) if isinstance(payload, dict) else None


def _normalised_title(text: str) -> str:
    return " ".join((text or "").lower().split())


def lookup_paper_by_title(title: str) -> dict | None:
    """
    Find the work whose title best matches ``title`` (used for year lookup).

    OpenAlex often holds several records for one paper — the original plus
    reindexed reprints carrying a much later publication_year. For staleness
    checks we want the date the work first appeared, so among records whose
    title actually matches we return the earliest year, breaking ties by
    citation count.
    """
    title = (title or "").strip()
    if len(title) < 8:
        return None

    params = _base_params()
    params["filter"] = f"title.search:{title}"
    params["per-page"] = 10
    payload = _client.get("/works", params, default={})
    results = payload.get("results", []) if isinstance(payload, dict) else []
    candidates = [_normalise(w) for w in results if isinstance(w, dict)]

    if not candidates:
        candidates = search_papers(title, limit=10)
    if not candidates:
        return None

    target = _normalised_title(title)
    exact = [c for c in candidates if _normalised_title(c["title"]) == target]
    pool = exact or candidates

    dated = [c for c in pool if isinstance(c.get("year"), int)]
    if not dated:
        return pool[0]

    return min(dated, key=lambda c: (c["year"], -(c.get("citationCount") or 0)))


def search_sota_for_task(task: str, metric: str, before_year: int, limit: int = 5) -> list[dict]:
    """
    Search for SOTA results on a task+metric published before a cutoff year.
    Used by the staleness checker to find baselines the paper should have
    compared against.
    """
    query = f"state of the art {task} {metric}".strip()
    return search_papers(query, limit=limit, year_range=f"-{before_year}")
