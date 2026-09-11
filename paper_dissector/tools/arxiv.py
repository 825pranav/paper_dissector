"""arXiv API wrapper — the last-resort literature backend.

arXiv needs no key and stays up when OpenAlex pauses anonymous search, so it is
the fallback that keeps the evidence stage alive. It covers preprints only (and
so is weak outside CS/physics/maths) and reports no citation counts, which is
why it is tried last rather than first.

Results are normalised into the Semantic Scholar record shape like every other
backend, so callers cannot tell the difference.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
import xml.etree.ElementTree as ET

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from paper_dissector import cache

log = logging.getLogger(__name__)

ARXIV_BASE = os.getenv("ARXIV_BASE", "https://export.arxiv.org/api/query")
# arXiv asks callers for roughly one request every three seconds.
_MIN_INTERVAL = float(os.getenv("ARXIV_MIN_INTERVAL", "3.0"))
_TIMEOUT = float(os.getenv("ARXIV_TIMEOUT", "40"))

_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
_MISS = object()

_rate_lock = threading.Lock()
_last_call = 0.0


class _Retryable(RuntimeError):
    """Transient arXiv failure."""


def _throttle() -> None:
    global _last_call
    with _rate_lock:
        elapsed = time.monotonic() - _last_call
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
        _last_call = time.monotonic()


def _clean(text: str) -> str:
    return " ".join((text or "").split())


def _sanitise_query(query: str) -> str:
    """arXiv's query parser dislikes punctuation; keep words and digits."""
    cleaned = re.sub(r"[^\w\s-]", " ", query or "")
    words = [w for w in cleaned.split() if len(w) > 1][:12]
    return " ".join(words)


def _year_clause(year_range: str | None) -> str | None:
    """Translate our 'YYYY-YYYY' / '-YYYY' / 'YYYY-' convention to a date filter."""
    if not year_range:
        return None
    value = year_range.strip()
    try:
        if value.startswith("-"):
            return f"submittedDate:[19910101 TO {int(value[1:]) - 1}1231]"
        if value.endswith("-"):
            return f"submittedDate:[{int(value[:-1])}0101 TO 20991231]"
        if "-" in value:
            start, end = value.split("-", 1)
            return f"submittedDate:[{int(start)}0101 TO {int(end)}1231]"
        year = int(value)
        return f"submittedDate:[{year}0101 TO {year}1231]"
    except ValueError:
        return None


@retry(
    retry=retry_if_exception_type(_Retryable),
    wait=wait_exponential(multiplier=2, min=2, max=20),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _fetch(params: dict) -> str:
    _throttle()
    try:
        resp = httpx.get(
            ARXIV_BASE, params=params,
            headers={"User-Agent": "paper-dissector"},
            timeout=_TIMEOUT, follow_redirects=True,
        )
    except httpx.HTTPError as exc:
        raise _Retryable(f"network error: {exc}") from exc

    if resp.status_code == 429 or resp.status_code >= 500:
        raise _Retryable(f"transient ({resp.status_code})")
    resp.raise_for_status()
    return resp.text


def _parse(xml_text: str) -> list[dict]:
    """Turn an Atom feed into Semantic Scholar shaped records."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        log.warning("could not parse arXiv response: %s", exc)
        return []

    out: list[dict] = []
    for entry in root.findall("atom:entry", _NS):
        title = _clean((entry.findtext("atom:title", default="", namespaces=_NS)))
        summary = _clean(entry.findtext("atom:summary", default="", namespaces=_NS))
        published = entry.findtext("atom:published", default="", namespaces=_NS)
        raw_id = entry.findtext("atom:id", default="", namespaces=_NS)

        year = None
        match = re.match(r"(\d{4})", published or "")
        if match:
            year = int(match.group(1))

        authors = [
            {"name": _clean(a.findtext("atom:name", default="", namespaces=_NS))}
            for a in entry.findall("atom:author", _NS)[:10]
        ]

        doi = entry.findtext("arxiv:doi", default="", namespaces=_NS) or None
        arxiv_id = (raw_id or "").rstrip("/").rsplit("/", 1)[-1]

        if not title:
            continue

        out.append({
            "paperId": f"arXiv:{arxiv_id}" if arxiv_id else "",
            "title": title,
            "year": year,
            "authors": authors,
            "abstract": summary,
            "citationCount": None,   # arXiv does not expose citation counts
            "url": raw_id or None,
            "externalIds": ({"DOI": doi} if doi else {"ArXiv": arxiv_id}) if (doi or arxiv_id) else {},
        })
    return out


def search_papers(query: str, limit: int = 10, year_range: str | None = None) -> list[dict]:
    """Search arXiv. Returns records in the Semantic Scholar shape; never raises."""
    cleaned = _sanitise_query(query)
    if not cleaned:
        return []

    clause = _year_clause(year_range)
    words = cleaned.split()

    # `all:a b c` ORs the terms, which returns loosely related work (a query for
    # "self attention encoder" came back with quantum-memory papers). Require
    # every term first, then loosen only if that is too strict to match.
    strict = " AND ".join(f"all:{w}" for w in words)
    loose = f"all:{cleaned}"

    for attempt, expr in enumerate((strict, loose)):
        if not expr:
            continue
        search = f"({expr}) AND {clause}" if clause else expr
        params = {
            "search_query": search,
            "start": 0,
            "max_results": min(max(limit, 1), 50),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        key = cache.make_key("arxiv", params)
        cached = cache.get(key, _MISS)
        if cached is not _MISS:
            if cached or attempt:
                return cached
            continue

        try:
            results = _parse(_fetch(params))
        except Exception as exc:
            log.warning("arXiv search failed, degrading gracefully: %s", exc)
            return []

        cache.set(key, results)
        if results:
            return results

    return []


def lookup_paper_by_title(title: str) -> dict | None:
    """Resolve a paper record from its title."""
    title = (title or "").strip()
    if len(title) < 8:
        return None
    cleaned = _sanitise_query(title)
    if not cleaned:
        return None

    try:
        xml_text = _fetch({
            "search_query": f'ti:"{cleaned}"',
            "start": 0, "max_results": 5,
        })
    except Exception as exc:
        log.warning("arXiv title lookup failed: %s", exc)
        return []  # noqa: RET504 - caller treats falsy as "not found"

    results = _parse(xml_text)
    if not results:
        return None

    target = " ".join(title.lower().split())
    exact = [r for r in results if " ".join(r["title"].lower().split()) == target]
    pool = exact or results
    dated = [r for r in pool if isinstance(r.get("year"), int)]
    return min(dated, key=lambda r: r["year"]) if dated else pool[0]


def search_sota_for_task(task: str, metric: str, before_year: int, limit: int = 5) -> list[dict]:
    """Find work on a task+metric published before a cutoff, for staleness checks."""
    query = f"{task} {metric}".strip()
    return search_papers(query, limit=limit, year_range=f"-{before_year}")
